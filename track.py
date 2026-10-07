"""Ultralytics YOLO26 실시간 객체 추적(Object Tracking).

실행:
    python track.py                     # 웹캠 0번
    python track.py video.mp4           # 동영상 파일
    python track.py "https://www.youtube.com/watch?v=..."  # 유튜브
종료: 창을 클릭한 뒤 q / ESC 키, 또는 창의 X 버튼
"""

import sys
import time
from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = "yolo26n.pt"       # n(가장 빠름) / s / m / l / x(가장 정확)
TRACKER = "bytetrack.yaml"      # 또는 "botsort.yaml"
DEFAULT_SOURCE = "0"            # 인자가 없을 때 쓸 소스 (기본 웹캠)
CONF_THRESHOLD = 0.5            # 이 신뢰도 이상인 객체만 추적
TRAIL_LENGTH = 30               # 이동 경로로 남길 최근 위치 개수
WINDOW_NAME = "YOLO26 Tracking"


def should_quit():
    """q/Q/ESC 키를 누르거나 창의 X 버튼으로 창을 닫으면 True."""
    key = cv2.waitKey(1) & 0xFF
    if key in (ord("q"), ord("Q"), 27):
        return True
    return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SOURCE
    if source.isdigit():
        source = int(source)  # 숫자는 웹캠 번호

    model = YOLO(MODEL_PATH)  # 처음 실행 시 가중치 자동 다운로드
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    trails = defaultdict(lambda: deque(maxlen=TRAIL_LENGTH))  # 추적 ID -> 최근 중심점들
    seen_ids = set()

    prev_time = time.time()
    try:
        # persist=True: 프레임 사이에서 같은 객체에 같은 ID를 유지
        for result in model.track(source, stream=True, persist=True, tracker=TRACKER,
                                  conf=CONF_THRESHOLD, verbose=False):
            annotated = result.plot()  # 상자 + "id:N 이름 신뢰도"

            if result.boxes.id is not None:
                ids = result.boxes.id.int().tolist()
                centers = result.boxes.xywh[:, :2].cpu().numpy()
                for track_id, (x, y) in zip(ids, centers):
                    seen_ids.add(track_id)
                    trails[track_id].append((int(x), int(y)))
                    points = np.array(trails[track_id], dtype=np.int32).reshape(-1, 1, 2)
                    cv2.polylines(annotated, [points], isClosed=False,
                                  color=(0, 255, 255), thickness=3)

            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            cv2.putText(annotated, f"FPS: {fps:.1f}  Total IDs: {len(seen_ids)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow(WINDOW_NAME, annotated)
            if should_quit():
                break
        else:
            # 영상이 끝나면 종료 키를 누를 때까지 창 유지
            while not should_quit():
                pass
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
