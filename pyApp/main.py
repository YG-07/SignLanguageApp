"""
“Good”手语手势识别阈值探究
手语传译官 - 手势规则识别程序（好手势）
任务：通过窗口调整两个阈值，让程序能准确识别"好"（👍）手势
打包：python -m PyInstaller --noconfirm --clean --onefile --windowed --name SignLanguageGood --collect-data mediapipe --exclude-module matplotlib --exclude-module mediapipe.model_maker --specpath .\build .\pyApp\main.py
"""

import queue
import sys
import threading
import tkinter as tk
import types
from tkinter import messagebox, ttk

import cv2


def install_frozen_matplotlib_stub():
    if not getattr(sys, "frozen", False):
        return

    # MediaPipe imports matplotlib for optional plotting helpers. This app only
    # needs OpenCV drawing, so a tiny stub keeps the windowed exe lightweight.
    matplotlib_stub = types.ModuleType("matplotlib")
    pyplot_stub = types.ModuleType("matplotlib.pyplot")
    matplotlib_stub.__path__ = []
    matplotlib_stub.pyplot = pyplot_stub
    pyplot_stub.__package__ = "matplotlib"
    sys.modules.setdefault("matplotlib", matplotlib_stub)
    sys.modules.setdefault("matplotlib.pyplot", pyplot_stub)


install_frozen_matplotlib_stub()

import mediapipe as mp


DEFAULT_THUMB_DIST_THRESHOLD = 0.35
DEFAULT_INDEX_DIST_THRESHOLD = 0.25
DEFAULT_MIDDLE_DIST_THRESHOLD = 0.25
DEFAULT_RING_DIST_THRESHOLD = 0.25
DEFAULT_PINKY_DIST_THRESHOLD = 0.25
DEFAULT_FINGER_DIST_THRESHOLD = 0.25
THRESHOLD_MIN = 0.00
THRESHOLD_MAX = 2.00
THRESHOLD_STEP = 0.01
THUMB_OPERATORS = ("<", "<=", "=", ">=", ">")
DEFAULT_THUMB_OPERATOR = "="
EQUAL_DISTANCE_TOLERANCE = 0.005

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def distance(p1, p2):
    return ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5


def format_threshold(value):
    return f"{value:.2f}"


def read_thresholds(thresholds, threshold_lock):
    with threshold_lock:
        return (
            thresholds["thumb"],
            thresholds["finger"],
            thresholds["thumb_operator"],
            thresholds["index"],
            thresholds["index_operator"],
            thresholds["middle"],
            thresholds["middle_operator"],
            thresholds["ring"],
            thresholds["ring_operator"],
            thresholds["pinky"],
            thresholds["pinky_operator"],
        )


def compare_thumb_distance(thumb_dist, operator, thumb_dist_threshold):
    if operator == "<":
        return thumb_dist < thumb_dist_threshold
    if operator == "<=":
        return thumb_dist <= thumb_dist_threshold
    if operator == "=":
        return abs(thumb_dist - thumb_dist_threshold) <= EQUAL_DISTANCE_TOLERANCE
    if operator == ">=":
        return thumb_dist >= thumb_dist_threshold
    if operator == ">":
        return thumb_dist > thumb_dist_threshold
    return False


def compare_index_distance(index_dist, operator, index_dist_threshold):
    if operator == "<":
        return index_dist < index_dist_threshold
    if operator == "<=":
        return index_dist <= index_dist_threshold
    if operator == "=":
        return abs(index_dist - index_dist_threshold) <= EQUAL_DISTANCE_TOLERANCE
    if operator == ">=":
        return index_dist >= index_dist_threshold
    if operator == ">":
        return index_dist > index_dist_threshold
    return False


def compare_middle_distance(middle_dist, operator, middle_dist_threshold):
    if operator == "<":
        return middle_dist < middle_dist_threshold
    if operator == "<=":
        return middle_dist <= middle_dist_threshold
    if operator == "=":
        return abs(middle_dist - middle_dist_threshold) <= EQUAL_DISTANCE_TOLERANCE
    if operator == ">=":
        return middle_dist >= middle_dist_threshold
    if operator == ">":
        return middle_dist > middle_dist_threshold
    return False


def compare_ring_distance(ring_dist, operator, ring_dist_threshold):
    if operator == "<":
        return ring_dist < ring_dist_threshold
    if operator == "<=":
        return ring_dist <= ring_dist_threshold
    if operator == "=":
        return abs(ring_dist - ring_dist_threshold) <= EQUAL_DISTANCE_TOLERANCE
    if operator == ">=":
        return ring_dist >= ring_dist_threshold
    if operator == ">":
        return ring_dist > ring_dist_threshold
    return False


def compare_pinky_distance(pinky_dist, operator, pinky_dist_threshold):
    if operator == "<":
        return pinky_dist < pinky_dist_threshold
    if operator == "<=":
        return pinky_dist <= pinky_dist_threshold
    if operator == "=":
        return abs(pinky_dist - pinky_dist_threshold) <= EQUAL_DISTANCE_TOLERANCE
    if operator == ">=":
        return pinky_dist >= pinky_dist_threshold
    if operator == ">":
        return pinky_dist > pinky_dist_threshold
    return False


def detect_good_gesture(landmarks, thresholds):
    (
        thumb_dist_threshold,
        finger_dist_threshold,
        thumb_operator,
        index_dist_threshold,
        index_operator,
        middle_dist_threshold,
        middle_operator,
        ring_dist_threshold,
        ring_operator,
        pinky_dist_threshold,
        pinky_operator,
    ) = thresholds

    wrist = landmarks[0]
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    # 计算拇指尖到手腕的距离
    thumb_dist = distance(thumb_tip, wrist)

    # 计算四指指尖到手腕的距离
    index_dist = distance(index_tip, wrist)
    middle_dist = distance(middle_tip, wrist)
    ring_dist = distance(ring_tip, wrist)
    pinky_dist = distance(pinky_tip, wrist)

    # 判断条件：5个手指的距离都满足各自的阈值条件
    if (
        compare_thumb_distance(thumb_dist, thumb_operator, thumb_dist_threshold)
        and compare_index_distance(index_dist, index_operator, index_dist_threshold)
        and compare_middle_distance(middle_dist, middle_operator, middle_dist_threshold)
        and compare_ring_distance(ring_dist, ring_operator, ring_dist_threshold)
        and compare_pinky_distance(pinky_dist, pinky_operator, pinky_dist_threshold)
    ):
        return "GOOD"
    return "UNKNOWN"


def print_startup_message():
    print("=" * 50)
    print("Sign Language Interpreter - Good Gesture Recognition")
    print("Use the threshold window to adjust values while running")
    print("Click the [X] button or press [Q] to close the recognition window")
    print("=" * 50)


def draw_landmark_ids(img, hand_landmarks):
    h, w, _ = img.shape
    mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    for landmark_id, landmark in enumerate(hand_landmarks.landmark):
        cx, cy = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(img, (cx, cy), 5, (0, 255, 0), -1)
        cv2.putText(
            img,
            str(landmark_id),
            (cx + 10, cy - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 0, 0),
            1,
        )


def draw_measurements(
    img,
    hand_landmarks,
    thumb_threshold,
    thumb_operator,
    finger_threshold,
    index_threshold,
    index_operator,
    middle_threshold,
    middle_operator,
    ring_threshold,
    ring_operator,
    pinky_threshold,
    pinky_operator,
):
    wrist = hand_landmarks.landmark[0]
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    middle_tip = hand_landmarks.landmark[12]
    ring_tip = hand_landmarks.landmark[16]
    pinky_tip = hand_landmarks.landmark[20]

    thumb_dist = distance(thumb_tip, wrist)
    index_dist = distance(index_tip, wrist)
    middle_dist = distance(middle_tip, wrist)
    ring_dist = distance(ring_tip, wrist)
    pinky_dist = distance(pinky_tip, wrist)
    max_finger_dist = max(index_dist, middle_dist, ring_dist, pinky_dist)

    # 显示测量值（黄色）
    # cv2.putText(
    #     img,
    #     f"[Measured] Thumb to Wrist = {thumb_dist:.3f}",
    #     (50, 180),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     0.6,
    #     (0, 255, 255),
    #     2,
    # )
    # cv2.putText(
    #     img,
    #     f"[Measured] Max Finger to Wrist = {max_finger_dist:.3f}",
    #     (50, 210),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     0.6,
    #     (0, 255, 255),
    #     2,
    # )

    # 显示当前设定的阈值（黄色）
    cv2.putText(
        img,
        f"[Your Setting] 4-0 Distance {thumb_operator} {thumb_threshold:.3f}",
        (50, 250),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1,
    )
    # cv2.putText(
    #     img,
    #     f"[Your Setting] Finger Threshold = {finger_threshold:.3f}",
    #     (50, 275),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     0.5,
    #     (0, 255, 255),
    #     1,
    # )

    # 动态提示：告诉学生阈值应该往哪个方向调
    if not compare_thumb_distance(thumb_dist, thumb_operator, thumb_threshold):
        cv2.putText(
            img,
            f"TIP: Try 4-0 distance = {thumb_dist:.3f}",
            (50, 310),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
        )
    if not compare_index_distance(index_dist, index_operator, index_threshold):
        cv2.putText(
            img,
            f"TIP: Try 8-0 distance = {index_dist:.3f}",
            (50, 335),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
        )
    if not compare_middle_distance(middle_dist, middle_operator, middle_threshold):
        cv2.putText(
            img,
            f"TIP: Try 12-0 distance = {middle_dist:.3f}",
            (50, 360),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
        )
    if not compare_ring_distance(ring_dist, ring_operator, ring_threshold):
        cv2.putText(
            img,
            f"TIP: Try 16-0 distance = {ring_dist:.3f}",
            (50, 385),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
        )
    if not compare_pinky_distance(pinky_dist, pinky_operator, pinky_threshold):
        cv2.putText(
            img,
            f"TIP: Try 20-0 distance = {pinky_dist:.3f}",
            (50, 410),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
        )
    # if max_finger_dist >= finger_threshold:
    #     cv2.putText(
    #         img,
    #         f"TIP: Try Finger Threshold = {max_finger_dist - 0.03:.3f}",
    #         (50, 335),
    #         cv2.FONT_HERSHEY_SIMPLEX,
    #         0.45,
    #         (0, 0, 255),
    #         1,
    #     )


def draw_status_text(img, result):
    color = (0, 255, 0) if result == "GOOD" else (0, 0, 255)
    cv2.putText(
        img,
        f"Result: {result}",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        color,
        3,
    )

    cv2.putText(
        img,
        "Make [GOOD] gesture (thumbs up)",
        (50, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
    )
    # cv2.putText(
    #     img,
    #     "Use the control window, then click OK",
    #     (50, 120),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     0.45,
    #     (0, 255, 255),
    #     1,
    # )
    # cv2.putText(
    #     img,
    #     "Click [X] or press [Q] to close",
    #     (50, 150),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     0.5,
    #     (0, 255, 0),
    #     1,
    # )


def run_recognition(thresholds, threshold_lock, stop_event, messages):
    cap = None
    hands = None
    window_name = "Gesture Recognition - Good Gesture"

    try:
        print_startup_message()

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("无法打开摄像头，请检查摄像头是否连接或被其他程序占用。")

        hands = mp_hands.Hands()
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        while not stop_event.is_set():
            success, img = cap.read()
            if not success:
                raise RuntimeError("无法读取摄像头画面。")

            (
                thumb_threshold,
                finger_threshold,
                thumb_operator,
                index_threshold,
                index_operator,
                middle_threshold,
                middle_operator,
                ring_threshold,
                ring_operator,
                pinky_threshold,
                pinky_operator,
            ) = read_thresholds(
                thresholds,
                threshold_lock,
            )
            current_thresholds = (
                thumb_threshold,
                finger_threshold,
                thumb_operator,
                index_threshold,
                index_operator,
                middle_threshold,
                middle_operator,
                ring_threshold,
                ring_operator,
                pinky_threshold,
                pinky_operator,
            )

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            result = "No Hand Detected"

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    draw_landmark_ids(img, hand_landmarks)
                    draw_measurements(
                        img,
                        hand_landmarks,
                        thumb_threshold,
                        thumb_operator,
                        finger_threshold,
                    )
                    result = detect_good_gesture(
                        hand_landmarks.landmark,
                        current_thresholds,
                    )

            draw_status_text(img, result)
            cv2.imshow(window_name, img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("Q"):
                stop_event.set()
                break

            try:
                window_visible = cv2.getWindowProperty(
                    window_name,
                    cv2.WND_PROP_VISIBLE,
                )
            except cv2.error:
                stop_event.set()
                break

            if window_visible < 1:
                stop_event.set()
                break
    except Exception as exc:
        messages.put(("error", str(exc)))
        stop_event.set()
    finally:
        if cap is not None:
            cap.release()
        if hands is not None:
            hands.close()
        cv2.destroyAllWindows()
        stop_event.set()


def create_control_window(thresholds, threshold_lock, stop_event, messages):
    root = tk.Tk()
    root.title("手势识别阈值调整")
    root.resizable(False, False)

    root.columnconfigure(0, weight=1)

    frame = ttk.Frame(root, padding=16)
    frame.grid(row=0, column=0, sticky="nsew")
    frame.columnconfigure(2, weight=1)

    thumb_var = tk.StringVar(value=format_threshold(DEFAULT_THUMB_DIST_THRESHOLD))
    thumb_operator_var = tk.StringVar(value=DEFAULT_THUMB_OPERATOR)
    index_var = tk.StringVar(value=format_threshold(DEFAULT_INDEX_DIST_THRESHOLD))
    index_operator_var = tk.StringVar(value=DEFAULT_THUMB_OPERATOR)
    middle_var = tk.StringVar(value=format_threshold(DEFAULT_MIDDLE_DIST_THRESHOLD))
    middle_operator_var = tk.StringVar(value=DEFAULT_THUMB_OPERATOR)
    ring_var = tk.StringVar(value=format_threshold(DEFAULT_RING_DIST_THRESHOLD))
    ring_operator_var = tk.StringVar(value=DEFAULT_THUMB_OPERATOR)
    pinky_var = tk.StringVar(value=format_threshold(DEFAULT_PINKY_DIST_THRESHOLD))
    pinky_operator_var = tk.StringVar(value=DEFAULT_THUMB_OPERATOR)
    status_var = tk.StringVar(value='调整数值后点击"确定"，识别窗口会立即使用新阈值。')
    window_closed = {"value": False}

    # 4号点到0号点的距离（拇指）
    ttk.Label(frame, text="4号点到0号点的距离").grid(
        row=0,
        column=0,
        sticky="w",
        padx=(0, 12),
        pady=(0, 10),
    )
    thumb_operator_combobox = ttk.Combobox(
        frame,
        textvariable=thumb_operator_var,
        values=THUMB_OPERATORS,
        width=4,
        state="readonly",
    )
    thumb_operator_combobox.grid(
        row=0,
        column=1,
        sticky="ew",
        padx=(0, 12),
        pady=(0, 10),
    )
    thumb_spinbox = tk.Spinbox(
        frame,
        from_=THRESHOLD_MIN,
        to=THRESHOLD_MAX,
        increment=THRESHOLD_STEP,
        format="%.2f",
        textvariable=thumb_var,
        width=8,
        justify="right",
    )
    thumb_spinbox.grid(row=0, column=2, sticky="ew", pady=(0, 10))

    # 8号点到0号点的距离（食指）
    ttk.Label(frame, text="8号点到0号点的距离").grid(
        row=1,
        column=0,
        sticky="w",
        padx=(0, 12),
        pady=(0, 10),
    )
    index_operator_combobox = ttk.Combobox(
        frame,
        textvariable=index_operator_var,
        values=THUMB_OPERATORS,
        width=4,
        state="readonly",
    )
    index_operator_combobox.grid(
        row=1,
        column=1,
        sticky="ew",
        padx=(0, 12),
        pady=(0, 10),
    )
    index_spinbox = tk.Spinbox(
        frame,
        from_=THRESHOLD_MIN,
        to=THRESHOLD_MAX,
        increment=THRESHOLD_STEP,
        format="%.2f",
        textvariable=index_var,
        width=8,
        justify="right",
    )
    index_spinbox.grid(row=1, column=2, sticky="ew", pady=(0, 10))

    # 12号点到0号点的距离（中指）
    ttk.Label(frame, text="12号点到0号点的距离").grid(
        row=2,
        column=0,
        sticky="w",
        padx=(0, 12),
        pady=(0, 10),
    )
    middle_operator_combobox = ttk.Combobox(
        frame,
        textvariable=middle_operator_var,
        values=THUMB_OPERATORS,
        width=4,
        state="readonly",
    )
    middle_operator_combobox.grid(
        row=2,
        column=1,
        sticky="ew",
        padx=(0, 12),
        pady=(0, 10),
    )
    middle_spinbox = tk.Spinbox(
        frame,
        from_=THRESHOLD_MIN,
        to=THRESHOLD_MAX,
        increment=THRESHOLD_STEP,
        format="%.2f",
        textvariable=middle_var,
        width=8,
        justify="right",
    )
    middle_spinbox.grid(row=2, column=2, sticky="ew", pady=(0, 10))

    # 16号点到0号点的距离（无名指）
    ttk.Label(frame, text="16号点到0号点的距离").grid(
        row=3,
        column=0,
        sticky="w",
        padx=(0, 12),
        pady=(0, 10),
    )
    ring_operator_combobox = ttk.Combobox(
        frame,
        textvariable=ring_operator_var,
        values=THUMB_OPERATORS,
        width=4,
        state="readonly",
    )
    ring_operator_combobox.grid(
        row=3,
        column=1,
        sticky="ew",
        padx=(0, 12),
        pady=(0, 10),
    )
    ring_spinbox = tk.Spinbox(
        frame,
        from_=THRESHOLD_MIN,
        to=THRESHOLD_MAX,
        increment=THRESHOLD_STEP,
        format="%.2f",
        textvariable=ring_var,
        width=8,
        justify="right",
    )
    ring_spinbox.grid(row=3, column=2, sticky="ew", pady=(0, 10))

    # 20号点到0号点的距离（小指）
    ttk.Label(frame, text="20号点到0号点的距离").grid(
        row=4,
        column=0,
        sticky="w",
        padx=(0, 12),
        pady=(0, 10),
    )
    pinky_operator_combobox = ttk.Combobox(
        frame,
        textvariable=pinky_operator_var,
        values=THUMB_OPERATORS,
        width=4,
        state="readonly",
    )
    pinky_operator_combobox.grid(
        row=4,
        column=1,
        sticky="ew",
        padx=(0, 12),
        pady=(0, 10),
    )
    pinky_spinbox = tk.Spinbox(
        frame,
        from_=THRESHOLD_MIN,
        to=THRESHOLD_MAX,
        increment=THRESHOLD_STEP,
        format="%.2f",
        textvariable=pinky_var,
        width=8,
        justify="right",
    )
    pinky_spinbox.grid(row=4, column=2, sticky="ew", pady=(0, 10))

    def parse_threshold(value, label):
        try:
            threshold = float(value)
        except ValueError as exc:
            raise ValueError(f"{label} 必须是数字。") from exc

        if not THRESHOLD_MIN <= threshold <= THRESHOLD_MAX:
            raise ValueError(f"{label} 必须在 {THRESHOLD_MIN:.2f} 到 {THRESHOLD_MAX:.2f} 之间。")

        return threshold

    def apply_thresholds():
        try:
            thumb_value = parse_threshold(thumb_var.get(), "4号点到0号点的距离")
            index_value = parse_threshold(index_var.get(), "8号点到0号点的距离")
            middle_value = parse_threshold(middle_var.get(), "12号点到0号点的距离")
            ring_value = parse_threshold(ring_var.get(), "16号点到0号点的距离")
            pinky_value = parse_threshold(pinky_var.get(), "20号点到0号点的距离")
        except ValueError as exc:
            messagebox.showerror("输入错误", str(exc), parent=root)
            return

        thumb_operator = thumb_operator_var.get()
        index_operator = index_operator_var.get()
        middle_operator = middle_operator_var.get()
        ring_operator = ring_operator_var.get()
        pinky_operator = pinky_operator_var.get()

        for op, name in [
            (thumb_operator, "4号点"),
            (index_operator, "8号点"),
            (middle_operator, "12号点"),
            (ring_operator, "16号点"),
            (pinky_operator, "20号点"),
        ]:
            if op not in THUMB_OPERATORS:
                messagebox.showerror("输入错误", f"请为{name}选择有效的判断符号。", parent=root)
                return

        with threshold_lock:
            thresholds["thumb"] = thumb_value
            thresholds["thumb_operator"] = thumb_operator
            thresholds["index"] = index_value
            thresholds["index_operator"] = index_operator
            thresholds["middle"] = middle_value
            thresholds["middle_operator"] = middle_operator
            thresholds["ring"] = ring_value
            thresholds["ring_operator"] = ring_operator
            thresholds["pinky"] = pinky_value
            thresholds["pinky_operator"] = pinky_operator

        thumb_var.set(format_threshold(thumb_value))
        index_var.set(format_threshold(index_value))
        middle_var.set(format_threshold(middle_value))
        ring_var.set(format_threshold(ring_value))
        pinky_var.set(format_threshold(pinky_value))
        status_var.set("已应用所有阈值设置")

    ok_button = ttk.Button(frame, text="确定", command=apply_thresholds)
    ok_button.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(4, 10))

    ttk.Label(frame, textvariable=status_var, foreground="#555555").grid(
        row=6,
        column=0,
        columnspan=3,
        sticky="w",
    )

    def close_window():
        if window_closed["value"]:
            return
        window_closed["value"] = True
        stop_event.set()
        root.destroy()

    def poll_messages():
        while True:
            try:
                level, text = messages.get_nowait()
            except queue.Empty:
                break

            status_var.set(text)
            if level == "error" and not window_closed["value"]:
                messagebox.showerror("运行错误", text, parent=root)

        if stop_event.is_set():
            close_window()
            return

        root.after(100, poll_messages)

    root.protocol("WM_DELETE_WINDOW", close_window)
    root.bind("<Return>", lambda _event: apply_thresholds())
    root.after(100, poll_messages)

    return root


def main():
    thresholds = {
        "thumb": DEFAULT_THUMB_DIST_THRESHOLD,
        "finger": DEFAULT_FINGER_DIST_THRESHOLD,
        "thumb_operator": DEFAULT_THUMB_OPERATOR,
        "index": DEFAULT_INDEX_DIST_THRESHOLD,
        "index_operator": DEFAULT_THUMB_OPERATOR,
        "middle": DEFAULT_MIDDLE_DIST_THRESHOLD,
        "middle_operator": DEFAULT_THUMB_OPERATOR,
        "ring": DEFAULT_RING_DIST_THRESHOLD,
        "ring_operator": DEFAULT_THUMB_OPERATOR,
        "pinky": DEFAULT_PINKY_DIST_THRESHOLD,
        "pinky_operator": DEFAULT_THUMB_OPERATOR,
    }
    threshold_lock = threading.Lock()
    stop_event = threading.Event()
    messages = queue.Queue()

    root = create_control_window(thresholds, threshold_lock, stop_event, messages)
    recognition_thread = threading.Thread(
        target=run_recognition,
        args=(thresholds, threshold_lock, stop_event, messages),
        daemon=True,
    )
    recognition_thread.start()

    try:
        root.mainloop()
    finally:
        stop_event.set()
        recognition_thread.join(timeout=3)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
