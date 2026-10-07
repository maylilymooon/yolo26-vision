"""나만의 제스처 데이터 수집기 (Tkinter UI).

화면의 손에서 랜드마크 21개를 뽑아 gesture_data/landmarks.csv 에 라벨과 함께 저장한다.
이미지가 아니라 좌표만 저장하므로 용량이 작고, 학습도 몇 초면 끝난다.

실행:
    python gesture_collect.py

사용 순서:
    1. 소스(웹캠 번호 / 영상 파일 / 유튜브 주소)를 넣고 [열기]
    2. 제스처 이름을 영문으로 입력하고 [추가] (예: ok_sign, rock, call_me)
    3. 목록에서 제스처를 고르고 [녹화] 또는 Space 키 → 손 모양을 유지하며 각도·거리를 조금씩 바꾸기
    4. 목표 개수가 차면 자동으로 멈춘다. 제스처마다 반복
    5. 다 모았으면 gesture_train.py 로 학습
"""

import re
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision
from PIL import Image, ImageTk

from custom_gesture_common import (HAND_MODEL_PATH, append_samples, landmarks_to_features,
                                   load_dataset, rewrite_dataset)
from hand_landmarker import download_youtube, draw_hands

DISPLAY_WIDTH = 800        # 화면에 보여줄 영상 너비 (px)
YOUTUBE_MAX_SIDE = 854     # 유튜브는 480p로 받기 (손 검출엔 충분하고 720p보다 훨씬 빨리 받아짐)
SAMPLES_PER_SEC = 10       # 녹화 중 초당 저장 개수 (비슷한 프레임이 너무 많이 쌓이지 않게)
DEFAULT_TARGET = 200       # 제스처 하나당 기본 목표 샘플 수
LABEL_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")  # OpenCV 화면에 한글이 안 나와서 영문만


class CollectorApp:
    def __init__(self, root):
        self.root = root
        root.title("제스처 데이터 수집기")
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.cap = None
        self.landmarker = None
        self.start_time = time.time()
        self.recording = False
        self.buffer = []           # 녹화 중 모은 (label, features)
        self.last_saved = 0.0
        self.sessions = []         # 되돌리기용: 녹화마다 저장한 개수
        self.frame_interval = 0    # 영상 파일이면 1/fps (원래 속도로 재생)
        self.next_due = time.time()

        X, y = load_dataset()
        self.counts = {}
        for label in y:
            self.counts[label] = self.counts.get(label, 0) + 1

        self.build_ui()
        self.refresh_labels()
        root.bind("<space>", self.on_space)
        self.loop()

    # ---------- 화면 구성 ----------
    def build_ui(self):
        self.video = ttk.Label(self.root, text="소스를 열어 주세요", anchor="center",
                               width=60, background="#222", foreground="#ddd")
        self.video.grid(row=0, column=0, rowspan=2, padx=8, pady=8, sticky="nsew")

        side = ttk.Frame(self.root, padding=8)
        side.grid(row=0, column=1, sticky="ns")

        src = ttk.LabelFrame(side, text="1. 입력 소스", padding=6)
        src.pack(fill="x")
        self.source_var = tk.StringVar(value="0")
        ttk.Entry(src, textvariable=self.source_var, width=32).pack(fill="x")
        ttk.Label(src, text="웹캠 번호(0, 1…) / 영상 파일 / 유튜브 주소", foreground="#666").pack(anchor="w")
        row = ttk.Frame(src)
        row.pack(fill="x", pady=(4, 0))
        self.mirror_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row, text="거울 모드(좌우 반전)", variable=self.mirror_var).pack(side="left")
        self.open_btn = ttk.Button(row, text="열기", command=self.open_source)
        self.open_btn.pack(side="right")

        lab = ttk.LabelFrame(side, text="2. 제스처 (라벨)", padding=6)
        lab.pack(fill="x", pady=8)
        row = ttk.Frame(lab)
        row.pack(fill="x")
        self.new_label_var = tk.StringVar()
        entry = ttk.Entry(row, textvariable=self.new_label_var, width=22)
        entry.pack(side="left", fill="x", expand=True)
        entry.bind("<Return>", lambda e: self.add_label())
        ttk.Button(row, text="추가", command=self.add_label).pack(side="right")
        self.label_list = tk.Listbox(lab, height=8, exportselection=False)
        self.label_list.pack(fill="x", pady=4)
        ttk.Button(lab, text="선택한 제스처 데이터 삭제", command=self.delete_label).pack(fill="x")

        rec = ttk.LabelFrame(side, text="3. 녹화", padding=6)
        rec.pack(fill="x")
        row = ttk.Frame(rec)
        row.pack(fill="x")
        ttk.Label(row, text="목표 개수").pack(side="left")
        self.target_var = tk.IntVar(value=DEFAULT_TARGET)
        ttk.Spinbox(row, from_=20, to=2000, increment=20, textvariable=self.target_var,
                    width=8).pack(side="right")
        self.record_btn = tk.Button(rec, text="● 녹화 (Space)", bg="#c62828", fg="white",
                                    font=("Malgun Gothic", 12, "bold"), command=self.toggle_record)
        self.record_btn.pack(fill="x", pady=6)
        self.progress = ttk.Progressbar(rec, maximum=DEFAULT_TARGET)
        self.progress.pack(fill="x")
        ttk.Button(rec, text="마지막 녹화 되돌리기", command=self.undo_last).pack(fill="x", pady=(6, 0))

        self.status_var = tk.StringVar(value="준비")
        ttk.Label(side, textvariable=self.status_var, wraplength=260, foreground="#1565c0").pack(
            fill="x", pady=8)

        tips = ("팁\n"
                "• 제스처마다 150~300개 정도 모으세요.\n"
                "• 녹화 중에 손을 조금씩 돌리고, 가까이·멀리 움직이세요.\n"
                "• 아무 제스처도 아닌 손 모양을 'none'으로 모아 두면\n  오인식이 줄어듭니다.")
        ttk.Label(side, text=tips, foreground="#555", justify="left").pack(fill="x")

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

    # ---------- 소스 열기 ----------
    def open_source(self):
        source = self.source_var.get().strip()
        if self.recording:
            self.toggle_record()
        if source.startswith("http"):
            self.open_btn.configure(state="disabled")  # 받는 중에 또 누르면 같은 파일을 동시에 받다가 깨짐
            self.status_var.set("유튜브 영상 받는 중… (처음 한 번만, 다음부턴 바로 열림)")
            threading.Thread(target=self._download_then_open, args=(source,), daemon=True).start()
        else:
            self._open(int(source) if source.isdigit() else source)

    def _download_then_open(self, url):
        def report(percent):
            text = f"유튜브 영상 받는 중… {percent:.0f}%" if percent < 100 else "받기 완료, 파일 정리 중…"
            self.root.after(0, self.status_var.set, text)

        try:
            path = download_youtube(url, max_side=YOUTUBE_MAX_SIDE, progress=report)
        except Exception as e:  # 네트워크·유튜브 오류를 화면에 표시
            self.root.after(0, self._download_failed, str(e))
            return
        self.root.after(0, self._open, path)

    def _download_failed(self, message):
        self.open_btn.configure(state="normal")
        self.status_var.set(f"다운로드 실패: {message}")

    def _open(self, source):
        self.open_btn.configure(state="normal")
        if self.cap is not None:
            self.cap.release()
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            self.status_var.set(f"소스를 열 수 없습니다: {source}")
            return
        self.cap = cap
        # 영상 파일은 원래 속도로 재생 (처리가 빨라서 그냥 두면 2배속 이상으로 지나감). 웹캠은 최대한 빨리.
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.frame_interval = 1 / fps if not isinstance(source, int) and fps > 0 else 0
        if self.landmarker is None:
            options = vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=1,  # 수집할 땐 한 손만: 두 손이 다 저장되면 라벨이 섞인다
            )
            self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.status_var.set(f"열림: {source}")

    def on_space(self, event):
        # 입력칸에 글자를 치는 중이면 녹화 단축키로 쓰지 않음
        if isinstance(event.widget, (tk.Entry, ttk.Entry)):
            return
        self.toggle_record()

    # ---------- 라벨 관리 ----------
    def refresh_labels(self):
        selected = self.selected_label()
        self.label_list.delete(0, "end")
        for name in sorted(self.counts):
            self.label_list.insert("end", f"{name}  ({self.counts[name]})")
        if selected in self.counts:
            idx = sorted(self.counts).index(selected)
            self.label_list.selection_set(idx)

    def selected_label(self):
        sel = self.label_list.curselection()
        if not sel:
            return None
        return self.label_list.get(sel[0]).rsplit("  (", 1)[0]

    def add_label(self):
        name = self.new_label_var.get().strip()
        if not LABEL_PATTERN.match(name):
            messagebox.showwarning("이름 확인", "제스처 이름은 영문·숫자·_ 만 쓸 수 있어요.\n예: ok_sign, rock, call_me")
            return
        self.counts.setdefault(name, 0)
        self.new_label_var.set("")
        self.refresh_labels()
        idx = sorted(self.counts).index(name)
        self.label_list.selection_clear(0, "end")
        self.label_list.selection_set(idx)

    def delete_label(self):
        name = self.selected_label()
        if name is None:
            return
        if not messagebox.askyesno("삭제", f"'{name}' 데이터 {self.counts[name]}개를 모두 지울까요?"):
            return
        X, y = load_dataset()
        keep = [i for i, label in enumerate(y) if label != name]
        rewrite_dataset(X[keep], [y[i] for i in keep])
        del self.counts[name]
        self.sessions.clear()  # 파일을 다시 썼으므로 되돌리기 기록은 무효
        self.refresh_labels()
        self.status_var.set(f"'{name}' 삭제됨")

    # ---------- 녹화 ----------
    def toggle_record(self):
        if not self.recording:
            label = self.selected_label()
            if label is None:
                messagebox.showinfo("제스처 선택", "목록에서 녹화할 제스처를 먼저 고르세요.")
                return
            if self.cap is None:
                messagebox.showinfo("소스", "먼저 입력 소스를 열어 주세요.")
                return
            self.recording = True
            self.rec_label = label
            self.buffer = []
            self.progress.configure(maximum=self.target_var.get(), value=0)
            self.record_btn.configure(text="■ 정지 (Space)", bg="#455a64")
            self.status_var.set(f"'{label}' 녹화 중… 손 모양을 유지하며 각도를 바꿔 보세요")
        else:
            self.recording = False
            self.record_btn.configure(text="● 녹화 (Space)", bg="#c62828")
            if self.buffer:
                append_samples(self.buffer)
                self.counts[self.rec_label] = self.counts.get(self.rec_label, 0) + len(self.buffer)
                self.sessions.append((self.rec_label, len(self.buffer)))
                self.status_var.set(f"'{self.rec_label}' {len(self.buffer)}개 저장 → {self.counts[self.rec_label]}개")
            else:
                self.status_var.set("손이 잡히지 않아 저장된 데이터가 없어요")
            self.buffer = []
            self.refresh_labels()

    def undo_last(self):
        if not self.sessions:
            self.status_var.set("되돌릴 녹화가 없어요 (이번 실행에서 녹화한 것만 되돌릴 수 있음)")
            return
        label, n = self.sessions.pop()
        X, y = load_dataset()
        rewrite_dataset(X[:-n], y[:-n])  # 마지막 녹화분은 파일 끝에 붙어 있다
        self.counts[label] -= n
        self.refresh_labels()
        self.status_var.set(f"'{label}' 마지막 녹화 {n}개를 지웠어요")

    # ---------- 영상 루프 ----------
    def loop(self):
        if self.cap is not None:
            ok, frame = self.cap.read()
            if not ok:  # 영상 파일이 끝나면 처음부터 다시
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.cap.read()
            if ok:
                self.process(frame)
        # 다음 프레임 예정 시각을 누적해서 계산: Windows 타이머 오차가 쌓여 느려지는 것을 막음
        now = time.time()
        interval = self.frame_interval if self.cap is not None else 0.03
        self.next_due = max(self.next_due + interval, now)
        self.root.after(max(1, int((self.next_due - now) * 1000)), self.loop)

    def process(self, frame):
        if self.mirror_var.get():
            frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp_ms = int((time.time() - self.start_time) * 1000)
        result = self.landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)
        draw_hands(frame, result)

        now = time.time()
        if self.recording and result.hand_landmarks and now - self.last_saved >= 1 / SAMPLES_PER_SEC:
            feats = landmarks_to_features(result.hand_landmarks[0],
                                          result.handedness[0][0].category_name)
            self.buffer.append((self.rec_label, feats))
            self.last_saved = now
            self.progress.configure(value=len(self.buffer))
            if len(self.buffer) >= self.target_var.get():
                self.toggle_record()

        if self.recording:
            cv2.circle(frame, (30, 30), 12, (0, 0, 255), -1)
            cv2.putText(frame, f"REC {self.rec_label} {len(self.buffer)}", (50, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif not result.hand_landmarks:
            cv2.putText(frame, "No hand", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)

        h, w = frame.shape[:2]
        scale = DISPLAY_WIDTH / w
        shown = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (DISPLAY_WIDTH, int(h * scale)))
        self.photo = ImageTk.PhotoImage(Image.fromarray(shown))  # 참조를 붙잡아 둬야 화면에 남음
        self.video.configure(image=self.photo, text="")

    def on_close(self):
        if self.recording:
            self.toggle_record()  # 녹화 중이던 데이터도 저장하고 닫기
        if self.cap is not None:
            self.cap.release()
        if self.landmarker is not None:
            self.landmarker.close()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    CollectorApp(root)
    root.mainloop()
