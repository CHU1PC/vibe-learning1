import atexit
import logging  # noqa
import os

logging.getLogger("streamlit").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
import cv2  # noqa
import numpy as np  # noqa
import streamlit as st  # noqa
from audio_utils import convert_and_cut_audio, cut_wav  # noqa  # 新しい関数も追加
from auth_utils import save_authenticated_user  # noqa
from auth_utils import (cosine_similarity, get_available_cameras,  # noqa
                        get_mean_embedding, load_authenticated_user,
                        load_registered_faces, register_face,
                        reset_authenticated_user)
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

# セッション状態でカメラの状態を管理
if "camera_active" not in st.session_state:
    st.session_state.camera_active = False
if "face_app" not in st.session_state:  # insightfaceのappをセッションで保持
    st.session_state.face_app = None
if "registered_faces" not in st.session_state:
    st.session_state.registered_faces = load_registered_faces()


if st.button("カメラで顔認証を開始/停止"):
    st.session_state.camera_active = not st.session_state.camera_active
    if st.session_state.camera_active and st.session_state.face_app is None:
        import insightface  # ここでインポート
        try:
            app = insightface.app.FaceAnalysis(
                providers=['CPUExecutionProvider'])  # CPUを指定
            app.prepare(ctx_id=-1, det_size=(640, 640))  # ctx_id=-1 for CPU
            st.session_state.face_app = app
            st.info("顔認証カメラを起動しました。下の画像フレームに顔を写してください。")
        except Exception as e:
            st.error(f"FaceAnalysisの初期化に失敗しました: {e}")
            st.session_state.camera_active = False  # エラー時はカメラを非アクティブに
    elif not st.session_state.camera_active:
        st.info("顔認証カメラを停止しました。")


if st.session_state.camera_active and st.session_state.face_app:
    img_file_buffer = st.camera_input("カメラ映像")

    if img_file_buffer is not None:
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8),
                               cv2.IMREAD_COLOR)

        app = st.session_state.face_app  # セッションからappを取得
        registered_faces = st.session_state.registered_faces  # セッションから登録顔情報を取得

        faces = app.get(cv2_img)
        processed_img = cv2_img.copy()  # 描画用のコピーを作成

        current_best_sim = 0.0
        current_best_name = None

        for idx, face in enumerate(faces):
            box = face.bbox.astype(int)
            cv2.rectangle(processed_img,
                          (box[0], box[1]),
                          (box[2], box[3]),
                          (0, 255, 0),
                          2)

            if registered_faces:
                sims = {
                    name: cosine_similarity(face.embedding,
                                            get_mean_embedding(emb_list))
                    for name, emb_list in registered_faces.items()
                }
                if sims:  # simsが空でないことを確認
                    best_name_local, best_sim_local = \
                        max(sims.items(), key=lambda x: x[1])
                    label = f"{best_name_local} ({best_sim_local:.2f})" \
                        if best_sim_local > 0.5 else "認証NG"

                    color = (0, 255, 0) if best_sim_local > 0.5\
                        else (0, 0, 255)
                    cv2.putText(processed_img, label,
                                (box[0], box[1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, color, 2)
                    if best_sim_local > current_best_sim:  # このフレームでの最高類似度を更新
                        current_best_sim = best_sim_local
                        current_best_name = best_name_local
            else:
                cv2.putText(processed_img, "未登録", (box[0], box[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        st.image(processed_img, channels="BGR", caption="認証中...")

        # 認証ボタン (十分な類似度の場合のみ表示)
        if current_best_name and current_best_sim >= 0.8:  # 閾値は適宜調整
            if st.button(f"{current_best_name}として認証する "
                         f"(スコア: {current_best_sim:.2f})", key="confirm_auth"):
                save_authenticated_user(current_best_name, current_best_sim)
                st.session_state["authenticated_user"] = current_best_name
                st.session_state["authenticated_score"] = current_best_sim
                st.success(f"認証成功: {current_best_name}"
                           f"（スコア: {current_best_sim:.2f}）")
                st.session_state.camera_active = False  # 認証成功したらカメラをオフ
                st.rerun()  # 画面を再描画して認証後の状態を表示

        # 顔登録セクション
        st.subheader("顔登録")
        if faces:
            face_options = [f"検出された顔 {i}" for i in range(len(faces))]
            if not face_options:
                st.write("登録対象の顔が検出されていません。")
            else:
                selected_face_idx_str = st.selectbox("登録する顔を選択:",
                                                     face_options,
                                                     key="face_select")

                if selected_face_idx_str:  # ユーザーが何かを選択した場合
                    selected_face_idx = \
                        int(selected_face_idx_str.split(" ")[-1])

                    registration_name = st.text_input("登録名を入力してください:",
                                                      key="reg_name")
                    if st.button("この顔を登録", key="register_btn"):
                        if (
                            registration_name and
                            0 <= selected_face_idx < len(faces)
                        ):

                            register_face(registration_name,
                                          faces[selected_face_idx].embedding)
                            st.session_state.registered_faces = \
                                load_registered_faces()  # 登録情報を更新
                            st.success(f"{registration_name} さんを登録しました。")
                            # 登録後に入力フィールドをクリアするために工夫が必要な場合がある
                        elif not registration_name:
                            st.warning("登録名を入力してください。")
                        else:
                            st.error("顔の選択が無効です。")
        else:
            st.info("カメラに顔を写すと、登録オプションが表示されます。")

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
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            if os.path.exists(processed_audio_path):
                os.remove(processed_audio_path)

else:
    st.warning("顔認証（精度80%以上）が必要です。カメラで認証してください。")
