from tensorflow import keras
import cv2
import mediapipe as mp
import numpy as np
import time
import csv
import pyautogui, sys
#size màn hiình
weight, height = pyautogui.size()

# tạo fps
pTime = 0
cTime = 0
label = ['moving', 'leftclick', 'rolldown', 'zoombig', 'zoomsmall', 'rollup', 'rightclick']
# Tạo bộ nhận dạng tay của Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
model = keras.models.load_model('my_model.h5')
# Mở webcam để bắt đầu quá trình theo dõi tay
cap = cv2.VideoCapture(0)
lcx = weight/2
lcy = height/2

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

while True:
    success, image = cap.read()
    # Thực hiện bộ nhận dạng tay của Mediapipe trên ảnh xám
    results = hands.process(image)
    h,w,c = image.shape
    # Nếu tay được phát hiện, vẽ các điểm trên các điểm đầu ngón tay và các kết nối giữa chúng
    if results.multi_hand_landmarks:
        output = []
        for hand_landmarks in results.multi_hand_landmarks:
            data = np.vstack([[s.x, s.y, s.z] for s in hand_landmarks.landmark])
            output.append(data)
            for id, lm in enumerate(hand_landmarks.landmark):
                if id == 5:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(image, (cx, cy), 15, (255,0,255), cv2.FILLED)
            # mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        pyautogui.moveTo(cx * (weight / w), cy * (height / h))
        output = np.array(output)
        predicted_class = np.argmax(model.predict(output))
        switch(predicted_class)
        # print(output)
        # print(type(output))
        # print(output.shape)

    #show fps
    cTime = time.time()
    fps = 1/(cTime-pTime)
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