import cv2
import mediapipe as mp
import numpy as np
import time
import csv

# tạo fps
pTime = 0
cTime = 0

# Tạo bộ nhận dạng tay của Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Mở webcam để bắt đầu quá trình theo dõi tay
cap = cv2.VideoCapture(0)


def madeCsv(output_data):
    #lưu file csv
    with open('zoomsmalltest.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

        # Ghi các giá trị vào file
        for row in output_data:
            writer.writerow(row)
output = []

while True:
    success, image = cap.read()
    # Thực hiện bộ nhận dạng tay của Mediapipe trên ảnh xám
    results = hands.process(image)
    # Nếu tay được phát hiện, vẽ các điểm trên các điểm đầu ngón tay và các kết nối giữa chúng
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # print(hand_landmarks)
            data = np.vstack([[s.x, s.y, s.z] for s in hand_landmarks.landmark])
            output.append(data)
            # for id, lm in enumerate(hand_landmarks.landmark):
            #     print(id, lm)
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    #show fps
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(image, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3,
                (255,0,255), 3)
    # Hiển thị kết quả
    cv2.imshow('Hand Tracking', image)

    if len(output) == 500: break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
madeCsv(output)
# Giải phóng bộ nhận dạng tay của Mediapipe và webcam
hands.close()
cap.release()
cv2.destroyAllWindows()