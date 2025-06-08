import atexit
import os
import warnings

warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import logging  # noqa

logging.getLogger("streamlit").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
import cv2  # noqa
import numpy as np  # noqa
import streamlit as st  # noqa
from audio_utils import convert_and_cut_audio, cut_wav  # noqa  # 新しい関数も追加
from auth_utils import save_authenticated_user  # noqa
from auth_utils import (cosine_similarity, get_mean_embedding,  # noqa
                        load_authenticated_user, load_registered_faces,
                        register_face, reset_authenticated_user)
# 各モジュールから必要な関数をインポート
from model_loaders import (get_asr_pipe, get_chatbot_pipe,  # noqa
                           get_easyocr_reader)

# --- 終了時に認証ファイルを削除するハンドラを登録 ---
atexit.register(reset_authenticated_user)

# --- 音声認識パイプライン ---
asr_pipe = get_asr_pipe()

# --- Streamlit UI ---
st.title("AI統合Webアプリ")

# --- やめるボタン ---
if st.button("やめる"):
    # セッションと認証ファイルをリセット
    st.session_state.pop("authenticated_user", None)
    st.session_state.pop("authenticated_score", None)
    reset_authenticated_user()
    st.success("認証状態をリセットしました。アプリを終了します。")
    st.stop()  # 以降の処理を停止

# --- 顔認証 ---
st.header("顔認証")
if st.button("カメラで顔認証を開始"):
    import insightface  # ここでインポートすることで、必要な時にだけロード

    app = insightface.app.FaceAnalysis()
    app.prepare(ctx_id=0, det_size=(640, 640))
    cap = cv2.VideoCapture(0)
    registered_faces = load_registered_faces()
    authenticated_user = None
    authenticated_score = 0.0

    st.info("ウィンドウが開いたら 's' で顔登録, 'q' で終了")
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("カメラが取得できません。")
            break
        faces = app.get(frame)
        for idx, face in enumerate(faces):
            box = face.bbox.astype(int)
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]),
                          (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{idx}", (box[0], box[1]-30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            if registered_faces:
                sims = {
                    name: cosine_similarity(face.embedding,
                                            get_mean_embedding(emb_list))
                    for name, emb_list in registered_faces.items()
                }
                best_name, best_sim = max(sims.items(), key=lambda x: x[1])

                label = f"{best_name} ({best_sim:.2f})" if best_sim > 0.5 \
                    else "認証NG"

                color = (0, 255, 0) if best_sim > 0.5 else (0, 0, 255)
                cv2.putText(frame, label, (box[0], box[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                if best_sim >= 0.8:
                    authenticated_user = best_name
                    authenticated_score = best_sim
            else:
                cv2.putText(frame, "未登録", (box[0], box[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.imshow('Face Authentication', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and faces:
            for idx, face in enumerate(faces):
                print(f"ID:{idx} 座標:{face.bbox.astype(int)}")
            try:
                select = int(input("登録したい顔のID番号を入力してください: "))
                if 0 <= select < len(faces):
                    name = input("登録する名前を入力してください: ")
                    register_face(name, faces[select].embedding)
                    registered_faces = load_registered_faces()
                else:
                    print("無効なIDです。")
            except Exception:
                print("IDの入力が正しくありません。")
    cap.release()
    cv2.destroyAllWindows()
    if authenticated_user and authenticated_score >= 0.8:
        save_authenticated_user(authenticated_user, authenticated_score)
        st.session_state["authenticated_user"] = authenticated_user
        st.session_state["authenticated_score"] = authenticated_score
        st.success(f"認証成功: {authenticated_user}"
                   f"（スコア: {authenticated_score:.2f}）")
    else:
        st.warning("認証に失敗しました。")

# --- 認証済みユーザーのみ利用可能な機能 ---
user, score = load_authenticated_user()
if user and score >= 0.8:
    st.success(f"認証成功: {user}（スコア: {score:.2f}）")

    # --- テキスト検出（EasyOCR） ---
    st.header("テキスト検出（EasyOCR）")

    uploaded_img = st.file_uploader("画像をアップロードしてください",
                                    type=["jpg", "jpeg", "png"],
                                    key="ocr_easy")
    if uploaded_img:
        img_array = np.frombuffer(uploaded_img.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        st.image(img, caption="アップロード画像", channels="BGR")

        reader = get_easyocr_reader()
        easyocr_result = reader.readtext(img)
        easyocr_texts = [text for _, text, _ in easyocr_result]
        st.write("EasyOCR検出結果:", easyocr_texts)

        # 画像への描画
        for bbox, text, conf in easyocr_result:
            pts = np.array(bbox, dtype=np.int32)
            cv2.polylines(img, [pts], isClosed=True,
                          color=(0, 255, 0), thickness=2)
            cv2.putText(img, text, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)
        st.image(img, caption="EasyOCR検出範囲", channels="BGR")

        chatbot_pipe = get_chatbot_pipe()
        input_text = "\n".join(easyocr_texts)
        messages = [
            {"role": "user", "content": input_text},
        ]
        chat_result = chatbot_pipe(
            messages,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7
        )
        for msg in chat_result[0]["generated_text"]:
            if msg["role"] == "assistant":
                st.write("チャットボットの返答:", msg["content"])

    # --- 音声→テキスト変換 ---
    st.header("音声→テキスト変換")

    audio_file = st.file_uploader("音声ファイルをアップロードしてください",
                                  type=["wav", "mp3", "m4a"])

    if audio_file:
        # アップロードされたファイルを一時的に保存
        temp_audio_path = "temp_uploaded_audio" +\
            os.path.splitext(audio_file.name)[-1]
        with open(temp_audio_path, "wb") as f:
            f.write(audio_file.read())

        processed_audio_path = "temp_audio_cut.wav"
        try:
            convert_and_cut_audio(temp_audio_path,
                                  processed_audio_path, max_sec=60)

            result = asr_pipe(processed_audio_path, return_timestamps=True)
            text = result["text"]
            st.write("音声認識結果:", text)

            # --- ここからchatbotに入力 ---
            chatbot_pipe = get_chatbot_pipe()
            messages = [
                {"role": "user", "content": text},
            ]
            chat_result = chatbot_pipe(
                messages,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7
            )
            for msg in chat_result[0]["generated_text"]:
                if msg["role"] == "assistant":
                    st.write("チャットボットの返答:", msg["content"])
        except ValueError as e:
            st.error(f"音声ファイルの処理エラー: {e}")
        finally:
            # 一時ファイルをクリーンアップ
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            if os.path.exists(processed_audio_path):
                os.remove(processed_audio_path)

else:
    st.warning("顔認証（精度80%以上）が必要です。カメラで認証してください。")
