import atexit
import contextlib
import os
import pickle
import warnings
import wave

warnings.filterwarnings("ignore")
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import logging

logging.getLogger("streamlit").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
import cv2
import insightface
import numpy as np
import streamlit as st
from transformers import pipeline

REGISTER_PATH = r"D:\program\programming\app\detect_techtrain\registered_faces.pkl"
AUTH_PATH = r"D:\program\programming\app\detect_techtrain\authenticated_user.txt"


# --- 終了時に認証ファイルを削除する ---
def reset_authenticated_user():
    if os.path.exists(AUTH_PATH):
        os.remove(AUTH_PATH)


# --- 顔認証ユーティリティ ---
def register_face(name, embedding):
    faces = load_registered_faces()
    if name in faces:
        faces[name].append(embedding)
    else:
        faces[name] = [embedding]
    with open(REGISTER_PATH, "wb") as f:
        pickle.dump(faces, f)


def get_mean_embedding(emb_list):
    return np.mean(emb_list, axis=0)


def load_registered_faces():
    if os.path.exists(REGISTER_PATH):
        with open(REGISTER_PATH, "rb") as f:
            faces = pickle.load(f)
        for k, v in faces.items():
            if not isinstance(v, list):
                faces[k] = [v]
        return faces
    return {}


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def save_authenticated_user(user, score):
    with open(AUTH_PATH, "w", encoding="utf-8") as f:
        f.write(f"{user},{score}")


def load_authenticated_user():
    if "authenticated_user" in st.session_state and "authenticated_score" in st.session_state:
        return st.session_state["authenticated_user"], st.session_state["authenticated_score"]
    if os.path.exists(AUTH_PATH):
        with open(AUTH_PATH, encoding="utf-8") as f:
            line = f.read().strip()
            if line:
                user, score = line.split(",")
                st.session_state["authenticated_user"] = user
                st.session_state["authenticated_score"] = float(score)
                return user, float(score)
    return None, 0.0


# --- 音声認識パイプライン ---
asr_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-small")

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
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{idx}", (box[0], box[1]-30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            if registered_faces:
                sims = {
                    name: cosine_similarity(face.embedding, get_mean_embedding(emb_list))
                    for name, emb_list in registered_faces.items()
                }
                best_name, best_sim = max(sims.items(), key=lambda x: x[1])
                label = f"{best_name} ({best_sim:.2f})" if best_sim > 0.5 else "認証NG"
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
        st.success(f"認証成功: {authenticated_user}（スコア: {authenticated_score:.2f}）")
    else:
        st.warning("認証に失敗しました。")

# --- 音声→テキスト機能 ---
user, score = load_authenticated_user()
if user and score >= 0.8:
    st.success(f"認証成功: {user}（スコア: {score:.2f}）")
    st.header("音声→テキスト変換")
    audio_file = st.file_uploader("音声ファイルをアップロードしてください", type=["wav", "mp3"])
    if audio_file:
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_file.read())

        def cut_wav(input_path, output_path, max_sec=60):
            with contextlib.closing(wave.open(input_path, 'rb')) as wf:
                framerate = wf.getframerate()
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                n_frames = wf.getnframes()
                duration = n_frames / float(framerate)
                cut_frames = int(min(duration, max_sec) * framerate)
                wf.rewind()
                frames = wf.readframes(cut_frames)
                with wave.open(output_path, 'wb') as out_wf:
                    out_wf.setnchannels(n_channels)
                    out_wf.setsampwidth(sampwidth)
                    out_wf.setframerate(framerate)
                    out_wf.writeframes(frames)
        if audio_file.name.lower().endswith('.wav'):
            cut_wav("temp_audio.wav", "temp_audio_cut.wav", max_sec=60)
            audio_path = "temp_audio_cut.wav"
        else:
            audio_path = "temp_audio.wav"
        result = asr_pipe(audio_path, return_timestamps=True)
        text = result["text"]
        st.write("音声認識結果:", text)

        # --- ここからchatbotに入力 ---
        # チャットボット用パイプライン
        chatbot_pipe = pipeline("text-generation", model="meta-llama/Llama-3.1-8B-Instruct")
        messages = [
            {"role": "user", "content": text},
        ]
        chat_result = chatbot_pipe(messages)
        # アシスタントの返答だけ表示
        for msg in chat_result[0]["generated_text"]:
            if msg["role"] == "assistant":
                st.write("チャットボットの返答:", msg["content"])
        atexit.register(reset_authenticated_user)
else:
    st.warning("顔認証（精度80%以上）が必要です。カメラで認証してください。")
