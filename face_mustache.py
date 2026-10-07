"""MediaPipe Face Landmarker로 얼굴에 콧수염 AR 필터 붙이기.

코와 윗입술 사이에 콧수염을 그리고, 입꼬리 간격으로 크기를,
두 입꼬리를 잇는 선의 기울기로 회전을 맞춘다 (고개를 기울여도 따라감).

실행:
    python face_mustache.py        # 웹캠 0번
    python face_mustache.py video.mp4
    python face_mustache.py "https://www.youtube.com/watch?v=FPXPxtBOS7E"

종료: 창을 클릭한 뒤 q / ESC 키, 또는 창의 X 버튼
"""

import math
import sys
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from hand_landmarker import download_youtube  # 유튜브 다운로드 재사용

MODEL_PATH = "face_landmarker.task"  # 얼굴 검출 + 478개 랜드마크 + 52개 blendshape
DEFAULT_SOURCE = "0"
NUM_FACES = 3                        # 동시에 콧수염을 붙일 최대 얼굴 수
WINDOW_NAME = "MediaPipe Face Mustache"
MUSTACHE_SCALE = 1.8                 # 입 너비 대비 콧수염 너비 배율

# Face Landmarker 478점 중 사용하는 인덱스
NOSE_BOTTOM = 2      # 코 밑
UPPER_LIP_TOP = 0    # 윗입술 중앙 위쪽
MOUTH_LEFT = 61      # 입꼬리 (화면 기준 왼쪽)
MOUTH_RIGHT = 291    # 입꼬리 (화면 기준 오른쪽)


def should_quit():
    """q/Q/ESC 키를 누르거나 창의 X 버튼으로 창을 닫으면 True."""
    key = cv2.waitKey(1) & 0xFF
    if key in (ord("q"), ord("Q"), 27):
        return True
    return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1


def make_mustache(width=400):
    """끝이 말려 올라간 콧수염 이미지를 BGRA(투명 배경)로 직접 그린다."""
    height = width // 3
    img = np.zeros((height, width, 4), np.uint8)
    color = (25, 30, 40, 255)  # 짙은 갈색 (B, G, R, A)
    cx = width / 2

    # 오른쪽 절반: t=0(가운데) → t=1(끝)로 가며 중심선은 살짝 처졌다가 끝에서 올라가고 두께는 얇아짐
    t = np.linspace(0, 1, 60)
    xs = cx + t * width * 0.40
    mid = height * 0.50 + np.sin(t * math.pi) * height * 0.15 - t ** 4 * height * 0.30
    thick = height * 0.50 * (1 - t) ** 0.7 + height * 0.06
    upper = mid - thick / 2 + np.exp(-t * 25) * height * 0.08  # 가운데(인중) 살짝 파임
    lower = mid + thick / 2
    right = np.concatenate([np.stack([xs, upper], 1), np.stack([xs, lower], 1)[::-1]])
    left = right.copy()
    left[:, 0] = 2 * cx - left[:, 0]  # 좌우 대칭

    for half in (right, left):
        cv2.fillPoly(img, [half.astype(np.int32)], color, cv2.LINE_AA)
    # 양 끝의 돌돌 말린 부분
    r = int(height * 0.09)
    for sign in (1, -1):
        end_x = cx + sign * (xs[-1] - cx)
        center = (int(end_x - sign * r * 0.3), int(mid[-1] - r))
        cv2.circle(img, center, r, color, max(int(height * 0.05), 2), cv2.LINE_AA)
    return img

def overlay_rgba(frame, overlay, center, angle_deg):
    """BGRA 이미지를 center 위치에 angle_deg만큼 회전시켜 알파 블렌딩한다."""
    h, w = overlay.shape[:2]
    # 회전해도 잘리지 않도록 대각선 크기의 정사각형 캔버스에서 회전
    side = int(math.hypot(w, h))
    canvas = np.zeros((side, side, 4), np.uint8)
    canvas[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = overlay
    m = cv2.getRotationMatrix2D((side / 2, side / 2), -angle_deg, 1.0)
    rotated = cv2.warpAffine(canvas, m, (side, side))

    x0, y0 = int(center[0] - side / 2), int(center[1] - side / 2)
    fh, fw = frame.shape[:2]
    # 화면 밖으로 나가는 부분은 잘라낸다
    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(fw, x0 + side), min(fh, y0 + side)
    if dx1 <= dx0 or dy1 <= dy0:
        return
    patch = rotated[sy0:sy0 + (dy1 - dy0), sx0:sx0 + (dx1 - dx0)]
    alpha = patch[:, :, 3:4].astype(np.float32) / 255.0
    roi = frame[dy0:dy1, dx0:dx1]
    roi[:] = (patch[:, :, :3] * alpha + roi * (1 - alpha)).astype(np.uint8)


def draw_mustaches(frame, result, mustache):
    """감지된 얼굴마다 콧수염을 위치·크기·기울기에 맞춰 붙인다."""
    h, w = frame.shape[:2]
    for landmarks in result.face_landmarks:
        def px(i):
            return np.array([landmarks[i].x * w, landmarks[i].y * h])

        left, right = px(MOUTH_LEFT), px(MOUTH_RIGHT)
        mouth_w = np.linalg.norm(right - left)
        if mouth_w < 5:
            continue
        center = px(NOSE_BOTTOM) * 0.4 + px(UPPER_LIP_TOP) * 0.6  # 코 밑과 윗입술 사이 (입술 쪽으로 조금)
        angle = math.degrees(math.atan2(right[1] - left[1], right[0] - left[0]))

        target_w = int(mouth_w * MUSTACHE_SCALE)
        resized = cv2.resize(mustache, (target_w, max(target_w // 3, 1)),
                             interpolation=cv2.INTER_AREA)
        overlay_rgba(frame, resized, center, angle)


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SOURCE
    if source.isdigit():
        source = int(source)
    elif source.startswith("http"):
        source = download_youtube(source)

    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=NUM_FACES,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"소스를 열 수 없습니다: {source}")
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    mustache = make_mustache()

    start = time.time()
    prev_time = start
    try:
        with vision.FaceLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = int((time.time() - start) * 1000)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                draw_mustaches(frame, result, mustache)

                now = time.time()
                fps = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now
                cv2.putText(frame, f"FPS: {fps:.1f}  Faces: {len(result.face_landmarks)}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow(WINDOW_NAME, frame)
                if should_quit():
                    break
            else:
                while not should_quit():
                    pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
