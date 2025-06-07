import os
import pickle

import cv2
import insightface
import numpy as np

REGISTER_PATH = "registered_faces.pkl"


def register_face(name, embedding):
    faces = load_registered_faces()
    # すでに登録があればリストに追加、なければ新規リスト
    if name in faces:
        faces[name].append(embedding)
    else:
        faces[name] = [embedding]
    with open(REGISTER_PATH, "wb") as f:
        pickle.dump(faces, f)
    print(f"{name} さんの顔を追加登録しました。")


def get_mean_embedding(emb_list):
    return np.mean(emb_list, axis=0)


def load_registered_faces():
    if os.path.exists(REGISTER_PATH):
        with open(REGISTER_PATH, "rb") as f:
            faces = pickle.load(f)
        # 既存データが1つだけの場合はリスト化
        for k, v in faces.items():
            if not isinstance(v, list):
                faces[k] = [v]
        return faces
    return {}


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


app = insightface.app.FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640, 640))

camera_index = 0
cap = cv2.VideoCapture(camera_index)
registered_faces = load_registered_faces()

frame_count = 0
faces = []


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % 3 == 0:
        faces = app.get(frame)

    # 顔ごとに番号を振って表示
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
        else:
            cv2.putText(frame, "未登録", (box[0], box[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    display_frame = cv2.resize(frame, None, fx=2, fy=1.5)

    cv2.imshow('Face Authentication', display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s') and faces:
        print(f"検出された顔の数: {len(faces)}")
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
        except Exception as e:
            print("IDの入力が正しくありません。")
    elif key == ord('c'):
        camera_index += 1
        cap.release()
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"カメラ index={camera_index} は存在しません。index=0 に戻します。")
            camera_index = 0
            cap = cv2.VideoCapture(camera_index)
        else:
            print(f"カメラを切り替えました（index={camera_index}）")

cap.release()
cv2.destroyAllWindows()
