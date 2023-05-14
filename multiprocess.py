from tensorflow import keras
import cv2
import mediapipe as mp
import numpy as np
import time
import joblib
import pyautogui
import math
import threading
import matplotlib.pyplot as plt
import queue
from multiprocessing import Value, Array

# Khởi tạo biến chia sẻ để theo dõi việc kết thúc chương trình
running = threading.Event()
running.set()

# Khởi tạo hàng đợi để truyền dữ liệu giữa các luồng
data_queue = queue.Queue(maxsize=2)

#size màn hiình
screen_width, screen_height = pyautogui.size()
frameR = 100
width = 640            # Width of Camera
height = 480            # Height of Camera

label = ['moving', 'leftclick', 'rolldown', 'zoombig', 'zoomsmall', 'rollup', 'rightclick']
# Tạo bộ nhận dạng tay của Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
model = joblib.load('SVMs_model.pkl')
# model = keras.models.load_model('my_model.h5')
# Mở webcam để bắt đầu quá trình theo dõi tay

def switch(lang):
    if lang == 1:
        pyautogui.click(button='left')
    elif lang == 2:
        pyautogui.scroll(-100)
    elif lang == 5:
        pyautogui.scroll(100)
    elif lang == 6:
        pyautogui.click(button='right')
    else:
        return "moving"
def smoothy(x,y,last_x,last_y, v):
    dis = math.sqrt((x-last_x)*(x-last_x)+(y-last_y)*(y-last_y))
    t = dis/v
    framec = t*30
    return framec + 1, t
frame_count = 0
fps = 1
h = 480
w = 640
pred = 0
def detect_hand():
    # tạo fps
    pTime = 0
    FPS = []
    v = 150
    fc = 2
    tm = 0.1
    hx = 0
    hy = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    while running.is_set():
        success, image = cap.read()
        # Thực hiện bộ nhận dạng tay của Mediapipe trên ảnh xám
        results = hands.process(image)
        # Nếu tay được phát hiện, vẽ các điểm trên các điểm đầu ngón tay và các kết nối giữa chúng
        if results.multi_hand_landmarks:
            output = []
            for hand_landmarks in results.multi_hand_landmarks:
                data = np.vstack([[s.x, s.y, s.z] for s in hand_landmarks.landmark])
                output.append(data)
                for id, lm in enumerate(hand_landmarks.landmark):
                    if id == 5:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        x3 = np.interp(cx, (frameR, width - frameR), (0, screen_width))
                        y3 = np.interp(cy, (frameR, height - frameR), (0, screen_height))
                        cv2.circle(image, (cx, cy), 15, (255,0,255), cv2.FILLED)
                # mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            fc, tm = smoothy(x3 * (width / w), y3 * (height / h), hx, hy, v)
            fc = int(fc)
            output = np.array(output)
            flattened_data = output.flatten()
            flattened_data = flattened_data.reshape(1, -1)
            predicted_class = model.predict(flattened_data)
            dt = cx, cy, fc, tm
            data_queue.put(dt)
            hx = x3 * (width / w)
            hy = y3 * (height / h)
            # predicted_class = np.argmax(model.predict(output))
            # switch(predicted_class)
            print(label[predicted_class[0]])
            # print(type(output))
            # print(output.shape)

        #show fps
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        FPS.append(fps)
        cv2.putText(image, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255,0,255), 3)
        # Hiển thị kết quả
        cv2.imshow('Hand Tracking', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # Vẽ biểu đồ
            cap.release()
            # plt.figure(figsize=(10, 6))
            # plt.plot(range(len(FPS)), FPS, label='FPS')
            # plt.xlabel('Frame')
            # plt.ylabel('FPS')
            # plt.title('FPS/Frame')
            # plt.legend()
            # plt.show()


def control_mouse():
    width, height = pyautogui.size()
    pyautogui.FAILSAFE = False
    while running.is_set():
        if not data_queue.empty():
            x3, y3, fc, tm = data_queue.get()
            if frame_count % fc == 0:
                if frame_count == 0:
                    fc = 2
                    tm = 0.1
                pyautogui.moveTo(x3 * (width / w), y3 * (height / h), duration=tm)
                switch(pred)
# Khởi tạo và chạy các luồng
if __name__ == '__main__':
    # Tạo và khởi chạy luồng cho bộ nhận dạng tay
    hand_thread = threading.Thread(target=detect_hand)

    # Tạo và khởi chạy luồng cho điều khiển chuộtq
    mouse_thread = threading.Thread(target=control_mouse)

    # Bắt đầu chạy các luồng
    hand_thread.start()
    mouse_thread.start()
    # Chờ người dùng nhấn phím để tắt các luồng
    if input() == '':
        running.clear()

    # Chờ cho đến khi tất cả các luồng kết thúc
    hand_thread.join()
    mouse_thread.join()

    # Giải phóng tài nguyên
    cv2.destroyAllWindows()
