"""나만의 제스처 분류기 학습기 (Tkinter UI).

gesture_collect.py 로 모은 gesture_data/landmarks.csv 를 읽어 분류 모델을 학습하고,
정확도·혼동 행렬을 보여준 뒤 gesture_data/custom_gesture_model.joblib 로 저장한다.

실행:
    python gesture_train.py
"""

import threading
import time
import tkinter as tk
from collections import Counter
from datetime import datetime
from tkinter import ttk

import joblib
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from custom_gesture_common import DATA_PATH, MODEL_OUT_PATH, load_dataset

matplotlib.rcParams["font.family"] = "Malgun Gothic"  # 그래프의 한글 깨짐 방지
MIN_PER_LABEL = 10   # 제스처 하나에 이보다 적으면 학습하지 않음

# 고를 수 있는 분류기. 입력이 숫자 63개뿐이라 모두 몇 초 안에 학습된다.
CLASSIFIERS = {
    "MLP 신경망 (추천)": lambda: MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=600,
                                            early_stopping=True, random_state=0),
    "랜덤 포레스트": lambda: RandomForestClassifier(n_estimators=200, random_state=0),
    "K-최근접 이웃 (KNN)": lambda: KNeighborsClassifier(n_neighbors=5, weights="distance"),
}


class TrainerApp:
    def __init__(self, root):
        self.root = root
        root.title("제스처 분류기 학습기")
        root.geometry("1100x720")
        self.build_ui()
        self.refresh_data()

    def build_ui(self):
        left = ttk.Frame(self.root, padding=8)
        left.pack(side="left", fill="y")

        data = ttk.LabelFrame(left, text="1. 데이터", padding=6)
        data.pack(fill="x")
        self.tree = ttk.Treeview(data, columns=("label", "count"), show="headings", height=10)
        self.tree.heading("label", text="제스처")
        self.tree.heading("count", text="샘플 수")
        self.tree.column("label", width=170)
        self.tree.column("count", width=80, anchor="e")
        self.tree.pack(fill="x")
        self.data_info = tk.StringVar()
        ttk.Label(data, textvariable=self.data_info, foreground="#555").pack(anchor="w", pady=4)
        ttk.Button(data, text="새로고침", command=self.refresh_data).pack(fill="x")

        opt = ttk.LabelFrame(left, text="2. 학습 설정", padding=6)
        opt.pack(fill="x", pady=8)
        ttk.Label(opt, text="분류기").pack(anchor="w")
        self.clf_var = tk.StringVar(value=next(iter(CLASSIFIERS)))
        ttk.Combobox(opt, textvariable=self.clf_var, values=list(CLASSIFIERS),
                     state="readonly").pack(fill="x")
        ttk.Label(opt, text="검증용 데이터 비율").pack(anchor="w", pady=(6, 0))
        self.test_var = tk.DoubleVar(value=0.2)
        row = ttk.Frame(opt)
        row.pack(fill="x")
        ttk.Scale(row, from_=0.1, to=0.4, variable=self.test_var,
                  command=lambda v: self.test_label.configure(text=f"{float(v):.0%}")).pack(
            side="left", fill="x", expand=True)
        self.test_label = ttk.Label(row, text="20%", width=5)
        self.test_label.pack(side="right")

        self.train_btn = tk.Button(left, text="▶ 학습 시작", bg="#2e7d32", fg="white",
                                   font=("Malgun Gothic", 12, "bold"), command=self.start_training)
        self.train_btn.pack(fill="x", pady=4)
        self.pbar = ttk.Progressbar(left, mode="indeterminate")
        self.pbar.pack(fill="x")
        self.result_var = tk.StringVar(value="")
        ttk.Label(left, textvariable=self.result_var, font=("Malgun Gothic", 11, "bold"),
                  foreground="#1565c0", wraplength=270).pack(fill="x", pady=8)

        right = ttk.Frame(self.root, padding=8)
        right.pack(side="left", fill="both", expand=True)
        self.fig = Figure(figsize=(5.5, 4.2), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.log = tk.Text(right, height=12, font=("Consolas", 10))
        self.log.pack(fill="x", pady=(8, 0))

    def write(self, text):
        self.log.insert("end", text + "\n")
        self.log.see("end")

    def refresh_data(self):
        self.X, self.y = load_dataset()
        counts = Counter(self.y)
        self.tree.delete(*self.tree.get_children())
        for label, n in sorted(counts.items()):
            tag = "low" if n < MIN_PER_LABEL else ""
            self.tree.insert("", "end", values=(label, n), tags=(tag,))
        self.tree.tag_configure("low", foreground="#c62828")
        self.data_info.set(f"제스처 {len(counts)}종 · 총 {len(self.y)}개\n파일: {DATA_PATH}")

    # ---------- 학습 ----------
    def start_training(self):
        self.refresh_data()
        counts = Counter(self.y)
        if len(counts) < 2:
            self.result_var.set("제스처가 2종류 이상 있어야 학습할 수 있어요.\ngesture_collect.py로 먼저 모아 주세요.")
            return
        low = [k for k, n in counts.items() if n < MIN_PER_LABEL]
        if low:
            self.result_var.set(f"샘플이 {MIN_PER_LABEL}개 미만인 제스처가 있어요: {', '.join(low)}")
            return
        self.train_btn.configure(state="disabled")
        self.pbar.start(10)
        self.result_var.set("학습 중…")
        self.log.delete("1.0", "end")
        # 학습은 별도 스레드에서: UI가 멈추지 않게
        threading.Thread(target=self.train, args=(self.clf_var.get(), self.test_var.get()),
                         daemon=True).start()

    def train(self, clf_name, test_size):
        X_tr, X_te, y_tr, y_te = train_test_split(self.X, self.y, test_size=test_size,
                                                  stratify=self.y, random_state=0)
        model = make_pipeline(StandardScaler(), CLASSIFIERS[clf_name]())
        t0 = time.time()
        model.fit(X_tr, y_tr)
        elapsed = time.time() - t0
        pred = model.predict(X_te)
        acc = accuracy_score(y_te, pred)
        labels = [str(label) for label in model.classes_]
        cm = confusion_matrix(y_te, pred, labels=labels)
        report = classification_report(y_te, pred, labels=labels, digits=3, zero_division=0)

        # 검증까지 끝났으니 전체 데이터로 다시 학습해서 저장 (데이터를 남김없이 활용)
        final = make_pipeline(StandardScaler(), CLASSIFIERS[clf_name]())
        final.fit(self.X, self.y)
        joblib.dump({
            "model": final,
            "labels": labels,
            "classifier": clf_name,
            "val_accuracy": acc,
            "num_samples": len(self.y),
            "trained_at": datetime.now().isoformat(timespec="seconds"),
        }, MODEL_OUT_PATH)

        self.root.after(0, self.show_result, clf_name, len(X_tr), len(X_te), elapsed,
                        acc, labels, cm, report)

    def show_result(self, clf_name, n_tr, n_te, elapsed, acc, labels, cm, report):
        self.pbar.stop()
        self.train_btn.configure(state="normal")
        self.result_var.set(f"검증 정확도 {acc:.1%}\n저장: {MODEL_OUT_PATH}")
        self.write(f"[{clf_name}] 학습 {n_tr}개 / 검증 {n_te}개 · {elapsed:.1f}초")
        self.write(report)
        self.write("→ 전체 데이터로 다시 학습해서 저장했어요. gesture_custom.py 로 실행해 보세요.")

        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("예측")
        ax.set_ylabel("정답")
        ax.set_title(f"혼동 행렬 (검증 정확도 {acc:.1%})")
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        self.fig.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    TrainerApp(root)
    root.mainloop()
