import os
import pickle

import cv2
import numpy as np
import streamlit as st

REGISTER_PATH = r"D:\program\programming\app\detect_techtrain"\
                r"\vibe-learning1\main\registered_faces.pkl"
AUTH_PATH = r"D:\program\programming\app\detect_techtrain\vibe-learning1"\
            r"\main\authenticated_user.txt"


def reset_authenticated_user():
    if os.path.exists(AUTH_PATH):
        os.remove(AUTH_PATH)


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
    if (
        "authenticated_user" in st.session_state and
        "authenticated_score" in st.session_state
    ):
        return (st.session_state["authenticated_user"],
                st.session_state["authenticated_score"])

    if os.path.exists(AUTH_PATH):
        with open(AUTH_PATH, encoding="utf-8") as f:
            line = f.read().strip()
            if line:
                user, score = line.split(",")
                st.session_state["authenticated_user"] = user
                st.session_state["authenticated_score"] = float(score)
                return user, float(score)
    return None, 0.0


# 利用可能なカメラを検出
def get_available_cameras(max_test=5):
    available = []
    for i in range(max_test):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available
