"""
“Good”手语手势识别阈值探究
手语传译官 - 手势规则识别程序（好手势）
任务：修改下面的两个阈值，让程序能准确识别"好"（👍）手势
"""

import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils


def distance(p1, p2):
    return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2)**0.5


# ==================== 代码修改区域 ====================
# 根据屏幕上显示的黄色数值，修改这两个数

thumb_dist_threshold = 0.35   # 拇指指尖到手腕的距离大于此值 → 拇指伸展开
finger_dist_threshold = 0.25   # 四指指尖到手腕的最大距离小于此值 → 四指弯曲

# ==================== 代码区域结束 ====================


def detect_good_gesture(landmarks):
    wrist = landmarks[0]
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    
    # 计算拇指尖到手腕的距离
    thumb_dist = distance(thumb_tip, wrist)
    
    # 计算四指指尖到手腕的距离，取最大值
    index_dist = distance(index_tip, wrist)
    middle_dist = distance(middle_tip, wrist)
    ring_dist = distance(ring_tip, wrist)
    pinky_dist = distance(pinky_tip, wrist)
    max_finger_dist = max(index_dist, middle_dist, ring_dist, pinky_dist)
    
    # 判断条件：拇指伸得远 AND 四指收得近
    if thumb_dist > thumb_dist_threshold and max_finger_dist < finger_dist_threshold:
        return "GOOD"
    return "UNKNOWN"


print("=" * 50)
print("Sign Language Interpreter - Good Gesture Recognition")
print("Task: Change two thresholds to recognize [GOOD] gesture")
print("Click the [X] button to close the window")
print("=" * 50)

cap = cv2.VideoCapture(0)

# 创建窗口并设置可关闭
window_name = "Gesture Recognition - Good Gesture"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

while True:
    success, img = cap.read()
    if not success:
        break
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    result = "No Hand Detected"
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            h, w, c = img.shape
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)
            
            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(img, (cx, cy), 5, (0, 255, 0), -1)
                cv2.putText(img, str(id), (cx + 10, cy - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            
            # 测量数值
            wrist = handLms.landmark[0]
            thumb_tip = handLms.landmark[4]
            index_tip = handLms.landmark[8]
            middle_tip = handLms.landmark[12]
            ring_tip = handLms.landmark[16]
            pinky_tip = handLms.landmark[20]
            
            thumb_dist = distance(thumb_tip, wrist)
            
            index_dist = distance(index_tip, wrist)
            middle_dist = distance(middle_tip, wrist)
            ring_dist = distance(ring_tip, wrist)
            pinky_dist = distance(pinky_tip, wrist)
            max_finger_dist = max(index_dist, middle_dist, ring_dist, pinky_dist)
            
            # 显示测量值（黄色）
            cv2.putText(img, f"[Measured] Thumb to Wrist = {thumb_dist:.3f}", (50, 180), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(img, f"[Measured] Max Finger to Wrist = {max_finger_dist:.3f}", (50, 210), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # 显示当前设定的阈值（青色）
            cv2.putText(img, f"[Your Setting] Thumb Threshold = {thumb_dist_threshold:.3f}", (50, 250), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(img, f"[Your Setting] Finger Threshold = {finger_dist_threshold:.3f}", (50, 275), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # 动态提示：告诉学生阈值应该往哪个方向调
            if thumb_dist <= thumb_dist_threshold:
                cv2.putText(img, f"TIP: Try Thumb Threshold = {thumb_dist + 0.03:.3f}", (50, 310), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            if max_finger_dist >= finger_dist_threshold:
                cv2.putText(img, f"TIP: Try Finger Threshold = {max_finger_dist - 0.03:.3f}", (50, 335), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            
            result = detect_good_gesture(handLms.landmark)
    
    # 显示结果：绿色=GOOD，红色=其他
    color = (0, 255, 0) if result == "GOOD" else (0, 0, 255)
    cv2.putText(img, f"Result: {result}", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    
    cv2.putText(img, "Make [GOOD] gesture (thumbs up)", (50, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(img, "Change the 2 thresholds in code, then rerun", (50, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    cv2.putText(img, "Click [X] to close", (50, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    cv2.imshow(window_name, img)
    
    # 检查是否点击了窗口的关闭按钮（X）
    # 方法1：检查按下的键是否是 'q'（保留供选择）
    # 方法2：检查窗口是否被关闭
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('Q'):  # 按 Q 也可以关闭
        break
    
    # 检查窗口是否还在（如果用户点了X，窗口会被销毁）
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
