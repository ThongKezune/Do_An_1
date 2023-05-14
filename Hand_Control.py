from tensorflow import keras
import cv2
import mediapipe as mp
import numpy as np
import time
import pyautogui
import joblib
import matplotlib.pyplot as plt
import math
# size màn hiình
screen_width, screen_height  = pyautogui.size()
frameR = 100
width = 640            # Width of Camera
height = 480
# tạo fps
pTime = 0
cTime = 0
label = ['moving', 'leftclick', 'rolldown', 'zoombig', 'zoomsmall', 'rollup', 'rightclick']
# Tạo bộ nhận dạng tay của Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
# model = keras.models.load_model('my_model.h5')
model = joblib.load('SVMs_model.pkl')
# Mở webcam để bắt đầu quá trình theo dõi tay
cap = cv2.VideoCapture(0)
value = []
def calculate_average(values):
    return sum(values) / len(values)
def smoothy(x,y,last_x,last_y, v):
    dis = math.sqrt((x-last_x)*(x-last_x)+(y-last_y)*(y-last_y))
    t = dis/v
    framec = t*30
    return framec + 1, t
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
fc = 2
tm = 0.1
hx = 0
hy = 0
FC = []
FPS = []
v = 150
frame_count = 0
while True:
    success, image = cap.read()
    # Thực hiện bộ nhận dạng tay của Mediapipe trên ảnh xám
    results = hands.process(image)
    frame_count += 1
    h,w,c = image.shape
    print(h, w)
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
        print(fc, tm)
        output = np.array(output)
        flattened_data = output.flatten()
        flattened_data = flattened_data.reshape(1, -1)
        predicted_class = model.predict(flattened_data)
        # predicted_class = np.argmax(model.predict(output))
        # if frame_count % fc == 0:
        pyautogui.moveTo(x3 * (width / w), y3 * (height / h))#, duration=tm)
        switch(predicted_class)
        hx = x3 * (width / w)
        hy = y3 * (height / h)
        # if frame_count % 100 == 0:
        #     v = v + 50
        #     fc = fc + 1
        # print(output)
        # print(type(output))
        # print(output.shape)

    #show fps
    cTime = time.time()
    fps = 1/(cTime-pTime)
    value.append(fps)
    if len(value) > 10:
        value = value[-10:]
    average = calculate_average(value)
    FPS.append(average)
    pTime = cTime
    cv2.putText(image, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3,
                (255,0,255), 3)
    # Hiển thị kết quả
    cv2.imshow('Hand Tracking', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Giải phóng bộ nhận dạng tay của Mediapipe và webcam
hands.close()
cap.release()
cv2.destroyAllWindows()
# Vẽ biểu đồ
plt.figure(figsize=(10, 6))
plt.plot(range(len(FPS)), FPS, label='FPS')
plt.xlabel('Frame')
plt.ylabel('FPS')
plt.title('FPS/Frame')
plt.legend()
plt.show()