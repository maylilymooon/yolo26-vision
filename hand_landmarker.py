"""MediaPipe Hand Landmarker 실시간 손 랜드마크 검출 (손 하나당 21개 관절점).

실행:
    python hand_landmarker.py        # 웹캠 0번
    python hand_landmarker.py 1      # 웹캠 1번
    python hand_landmarker.py video.mp4
    python hand_landmarker.py "https://www.youtube.com/shorts/PWGWHPTTJ80"  # 유튜브 (videos/ 에 받아서 재생)

종료: 창을 클릭한 뒤 q / ESC 키, 또는 창의 X 버튼
"""

import os
import re
import sys
import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

MODEL_PATH = "hand_landmarker.task"  # 손바닥 검출 + 랜드마크 모델 번들 (float16)
DEFAULT_SOURCE = "0"                 # 인자가 없을 때 쓸 소스 (기본 웹캠)
NUM_HANDS = 2                        # 동시에 검출할 최대 손 개수
WINDOW_NAME = "MediaPipe Hand Landmarker"
VIDEO_DIR = "videos"                 # 유튜브 영상 저장 폴더

CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS
FINGERTIPS = (4, 8, 12, 16, 20)      # 엄지/검지/중지/약지/새끼 끝


def should_quit():
    """q/Q/ESC 키를 누르거나 창의 X 버튼으로 창을 닫으면 True."""
    key = cv2.waitKey(1) & 0xFF
    if key in (ord("q"), ord("Q"), 27):
        return True
    return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1


def download_youtube(url, max_side=1280, progress=None):
    """yt-dlp로 유튜브 영상을 받아 로컬 경로를 반환 (이미 받았으면 재사용).

    max_side: 가로·세로 최대 픽셀 (1280 = 720p, 854 = 480p). 작을수록 빨리 받아진다.
    progress: 진행률(0~100)을 받을 함수. UI에 다운로드 상황을 보여줄 때 사용.
    """
    import imageio_ffmpeg
    import yt_dlp

    name = "%(id)s" if max_side == 1280 else f"%(id)s_{max_side}"
    # 주소에서 영상 ID를 바로 뽑아 이미 받은 파일이 있으면 즉시 반환 (유튜브 조회에 몇 초 걸려서)
    m = re.search(r"(?:v=|youtu\.be/|shorts/|embed/)([\w-]{11})", url)
    if m:
        cached = os.path.join(VIDEO_DIR, name.replace("%(id)s", m.group(1)) + ".mp4")
        if os.path.exists(cached):
            return cached

    video = f"bestvideo[ext=mp4][vcodec^=avc1][width<={max_side}][height<={max_side}]"
    opts = {
        # OpenCV가 읽기 쉬운 H.264(mp4) 영상만 (검출엔 소리 불필요)
        # HLS(m3u8)를 우선: 일반(DASH) 다운로드는 긴 영상에서 중간부터 깨진 파일이 받아지는 경우가 있음
        "format": f"{video}[protocol^=m3u8]/{video}/best[ext=mp4]/best",
        "outtmpl": os.path.join(VIDEO_DIR, name + ".%(ext)s"),
        "ffmpeg_location": imageio_ffmpeg.get_ffmpeg_exe(),  # HLS 조각 합치기용 (pip 패키지에 포함된 ffmpeg)
        "quiet": True,
        "noprogress": progress is not None,  # UI가 진행률을 보여줄 땐 터미널 출력 생략
    }
    if progress is not None:
        def hook(d):
            if d["status"] == "downloading":
                total = d.get("total_bytes") or d.get("total_bytes_estimate")
                if total:
                    progress(min(100.0, d["downloaded_bytes"] / total * 100))
        opts["progress_hooks"] = [hook]
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
        path = ydl.prepare_filename(info)
        if not os.path.exists(path):
            print(f"다운로드 중: {url}")
            ydl.process_ie_result(info, download=True)  # 위에서 조회한 정보로 바로 받기 (재조회 안 함)
    return path


def draw_hands(frame, result):
    """검출된 손마다 뼈대 선, 관절점, 왼손/오른손 라벨을 그린다."""
    h, w = frame.shape[:2]
    for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
        # 정규화 좌표(0~1)를 픽셀 좌표로 변환
        points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

        for conn in CONNECTIONS:
            cv2.line(frame, points[conn.start], points[conn.end], (255, 255, 255), 2)
        for i, pt in enumerate(points):
            color = (0, 0, 255) if i in FINGERTIPS else (0, 255, 0)
            cv2.circle(frame, pt, 5, color, -1)

        # MediaPipe는 좌우 반전(셀카)된 입력을 가정하므로, 반전 안 한 영상에선 Left/Right가 실제와 반대
        label = f"{handedness[0].category_name} {handedness[0].score:.2f}"
        x, y = points[0]  # 손목
        cv2.putText(frame, label, (x - 30, y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SOURCE
    if source.isdigit():
        source = int(source)  # 숫자면 웹캠 번호
    elif source.startswith("http"):
        source = download_youtube(source)

    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,  # 프레임 단위 동기 처리 (이전 프레임으로 추적)
        num_hands=NUM_HANDS,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"소스를 열 수 없습니다: {source}")
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    start = time.time()
    prev_time = start
    try:
        with vision.HandLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                # MediaPipe는 RGB 입력, VIDEO 모드는 단조 증가하는 ms 타임스탬프 필요
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = int((time.time() - start) * 1000)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                draw_hands(frame, result)

                now = time.time()
                fps = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now
                cv2.putText(frame, f"FPS: {fps:.1f}  Hands: {len(result.hand_landmarks)}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow(WINDOW_NAME, frame)
                if should_quit():
                    break
            else:
                # 영상이 끝나면 종료 키를 누를 때까지 마지막 프레임 유지
                while not should_quit():
                    pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
