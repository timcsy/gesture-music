"""辨識鏡頭前單手比出的台灣數字手勢（0–9）。"""

from __future__ import annotations

import time
import urllib.request
import math
import os
from pathlib import Path

import cv2
import mediapipe as mp


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")
GESTURE_NAMES = {
    0: "握拳",
    1: "食指伸出",
    2: "食指與中指伸出",
    3: "食指、中指、無名指伸出，拇指與小指收起",
    4: "四指伸出、拇指收起",
    5: "張開手掌",
    6: "拇指與小指伸出",
    7: "拇指與食指比出手槍形",
    8: "拇指、食指與中指伸出",
    9: "拇指、食指、中指、無名指伸出，小指收起",
}
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


def download_model() -> None:
    if MODEL_PATH.exists():
        return
    print("首次執行，正在下載手部辨識模型…")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception:
        MODEL_PATH.unlink(missing_ok=True)
        raise
    print(f"模型已儲存：{MODEL_PATH.name}")


def distance(a, b) -> float:
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def recognize_number(points) -> int | None:
    """辨認台灣常見單手數字手勢；不是單純計算伸出的手指根數。"""
    wrist = points[0]
    # 以指尖到手腕的距離和關節距離判斷手指是否伸直，較不依賴畫面方向。
    palm_width = max(distance(points[5], points[17]), 0.001)
    thumb_span = distance(points[4], points[5]) / palm_width
    thumb_reach = distance(points[4], wrist) / max(distance(points[3], wrist), 0.001)

    # 用拇指 IP 關節的彎曲角度分辨「拇指橫收在掌心」的 4 和「拇指張開」的 5。
    thumb_a = (points[2].x - points[3].x, points[2].y - points[3].y)
    thumb_b = (points[4].x - points[3].x, points[4].y - points[3].y)
    thumb_denominator = max(
        (thumb_a[0] ** 2 + thumb_a[1] ** 2) ** 0.5
        * (thumb_b[0] ** 2 + thumb_b[1] ** 2) ** 0.5,
        1e-6,
    )
    thumb_cosine = max(-1.0, min(1.0, (thumb_a[0] * thumb_b[0] + thumb_a[1] * thumb_b[1]) / thumb_denominator))
    thumb_angle = math.degrees(math.acos(thumb_cosine))

    # 留出灰區：拇指半收半開時不硬猜，降低 4/5 互相誤判。
    if thumb_angle >= 150 and thumb_span >= 0.52 and thumb_reach >= 1.02:
        thumb_state = True
    elif thumb_angle <= 135 or thumb_span <= 0.42 or thumb_reach <= 0.96:
        thumb_state = False
    else:
        thumb_state = None

    extended = {
        "thumb": thumb_state is True,
        "index": distance(points[8], wrist) > distance(points[6], wrist) * 1.10,
        "middle": distance(points[12], wrist) > distance(points[10], wrist) * 1.10,
        "ring": distance(points[16], wrist) > distance(points[14], wrist) * 1.08,
        "pinky": distance(points[20], wrist) > distance(points[18], wrist) * 1.06,
    }
    t, i, m, r, p = (extended[key] for key in ("thumb", "index", "middle", "ring", "pinky"))

    # 沿掌骨方向計算拇指在掌心內側或外側的位置，避免依賴畫面左右方向。
    axis_x = points[17].x - points[5].x
    axis_y = points[17].y - points[5].y
    axis_length = max((axis_x * axis_x + axis_y * axis_y) ** 0.5, 0.001)
    axis_x, axis_y = axis_x / axis_length, axis_y / axis_length

    def lateral(point):
        return (point.x - points[5].x) * axis_x + (point.y - points[5].y) * axis_y

    thumb_lateral = lateral(points[4]) / palm_width

    # 依使用者提供的台灣日常手勢範例：3 是食指、中指、無名指伸出，
    # 拇指橫收在掌內；9 則同樣三指伸出，但拇指向掌外張開。
    if i and m and r and not p and thumb_lateral > 0.20:
        return 3
    if i and m and r and not p and thumb_lateral < -0.20:
        return 9

    # 交叉食指與中指也保留為 9 的另一種常見比法。沿掌骨軸比較兩指
    # 近端和指尖的左右順序，手掌傾斜時比直接比較畫面 x 座標穩定。
    base_order = lateral(points[6]) - lateral(points[10])
    tip_order = lateral(points[8]) - lateral(points[12])
    crossed_index_middle = (
        base_order * tip_order < 0
        and abs(base_order) > palm_width * 0.025
        and abs(tip_order) > palm_width * 0.04
        and i and m
    )

    # 特殊形狀優先，避免把交叉手指誤認為一般伸指組合。
    if crossed_index_middle and not r and not p:
        return 9
    if t and i and m and not r and not p:
        return 8
    if t and i and not m and not r and not p:
        return 7
    if t and p and not i and not m and not r:
        return 6
    if thumb_state is True and i and m and r and p:
        return 5
    if thumb_state is False and i and m and r and p:
        return 4
    if not t and i and m and not r and not p:
        return 2
    if not t and i and not m and not r and not p:
        return 1
    if thumb_state is False and not i and not m and not r and not p:
        return 0
    return None


def main() -> None:
    download_model()
    camera = (
        cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if os.name == "nt"
        else cv2.VideoCapture(0)
    )
    if not camera.isOpened():
        raise RuntimeError("無法開啟相機。請確認相機已連接，且沒有被其他程式占用。")

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.55,
        min_hand_presence_confidence=0.55,
        min_tracking_confidence=0.5,
    )

    print("相機已啟動。按 Q 或 Esc 關閉。")
    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
            started_at = time.monotonic()
            while True:
                ok, frame = camera.read()
                if not ok:
                    print("讀取相機畫面失敗。")
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = int((time.monotonic() - started_at) * 1000)
                result = landmarker.detect_for_video(image, timestamp_ms)

                if result.hand_landmarks:
                    points = result.hand_landmarks[0]
                    number = recognize_number(points)
                    height, width = frame.shape[:2]
                    for start, end in CONNECTIONS:
                        a, b = points[start], points[end]
                        cv2.line(frame, (int(a.x * width), int(a.y * height)),
                                 (int(b.x * width), int(b.y * height)), (120, 240, 190), 3)
                    for point in points:
                        cv2.circle(frame, (int(point.x * width), int(point.y * height)), 5, (255, 255, 255), -1)
                    if number is None:
                        label = "Unknown gesture"
                        color = (170, 180, 195)
                    else:
                        label = f"Number: {number}  {GESTURE_NAMES[number]}"
                        color = (120, 240, 190)
                    cv2.putText(frame, label, (24, 58), cv2.FONT_HERSHEY_SIMPLEX,
                                0.85, color, 3, cv2.LINE_AA)
                else:
                    cv2.putText(frame, "Show one hand to the camera", (24, 58),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 225, 235), 2, cv2.LINE_AA)

                cv2.imshow("Hand Gesture Number Recognition", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError) as error:
        print(f"錯誤：{error}")
