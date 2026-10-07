"""나만의 제스처 수집·학습·추론 스크립트가 함께 쓰는 설정과 함수.

손 랜드마크 21개 점을 "손 위치·크기·왼오른손과 상관없는" 숫자 63개로 바꾼다.
같은 함수를 수집 때와 추론 때 똑같이 써야 학습한 모델이 제대로 동작한다.
"""

import csv
import os

import numpy as np

HAND_MODEL_PATH = "hand_landmarker.task"     # 랜드마크 추출용 (9교시 모델)
DATA_PATH = os.path.join("gesture_data", "landmarks.csv")
MODEL_OUT_PATH = os.path.join("gesture_data", "custom_gesture_model.joblib")

NUM_POINTS = 21
FEATURE_COLUMNS = [f"{axis}{i}" for i in range(NUM_POINTS) for axis in ("x", "y", "z")]


def landmarks_to_features(landmarks, handedness_name):
    """랜드마크 21개 → 정규화된 특징 벡터 63개.

    1. 손목(0번)을 원점으로 옮긴다      → 화면 어디에 손이 있든 같은 값
    2. 손목에서 가장 먼 점까지 거리로 나눈다 → 손이 크든 작든(가깝든 멀든) 같은 값
    3. 왼손이면 x를 뒤집는다            → 왼손·오른손을 같은 제스처로 취급
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    pts -= pts[0]
    scale = np.max(np.linalg.norm(pts[:, :2], axis=1))
    if scale > 0:
        pts /= scale
    if handedness_name == "Left":
        pts[:, 0] *= -1
    return pts.flatten()


def load_dataset(path=DATA_PATH):
    """CSV를 읽어 (특징 배열 X, 라벨 리스트 y)를 돌려준다. 파일이 없으면 빈 값."""
    if not os.path.exists(path):
        return np.empty((0, len(FEATURE_COLUMNS)), np.float32), []
    X, y = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            y.append(row["label"])
            X.append([float(row[c]) for c in FEATURE_COLUMNS])
    return np.array(X, np.float32).reshape(-1, len(FEATURE_COLUMNS)), y


def append_samples(rows, path=DATA_PATH):
    """[(label, features), ...] 를 CSV 끝에 덧붙인다 (처음이면 헤더도 쓴다)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new_file = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["label"] + FEATURE_COLUMNS)
        for label, feats in rows:
            writer.writerow([label] + [f"{v:.5f}" for v in feats])


def rewrite_dataset(X, y, path=DATA_PATH):
    """데이터 전체를 다시 쓴다 (라벨 삭제·되돌리기용)."""
    if os.path.exists(path):
        os.remove(path)
    if len(y):
        append_samples(list(zip(y, X)), path)
