# import cv2 # 웹캠 제어 및 ML 사용 
# import mediapipe as mp # 손 인식을 할 것
# import numpy as np

# max_num_hands = 2 # 손은 최대 1개만 인식
# gesture = { # **11가지나 되는 제스처 라벨, 각 라벨의 제스처 데이터는 이미 수집됨 (제스처 데이터 == 손가락 관절의 각도, 각각의 라벨)**
#     0:'fist', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five',
#     6:'six', 7:'rock', 8:'spiderman', 9:'yeah', 10:'ok',
# }
# rps_gesture = {0:'rock', 5:'paper', 9:'scissors'} # 우리가 사용할 제스처 라벨만 가져옴 

# # MediaPipe hands model
# mp_hands = mp.solutions.hands # 웹캠 영상에서 손가락 마디와 포인트를 그릴 수 있게 도와주는 유틸리티1
# mp_drawing = mp.solutions.drawing_utils # 웹캠 영상에서 손가락 마디와 포인트를 그릴 수 있게 도와주는 유틸리티2

#  # 손가락 detection 모듈을 초기화
# hands = mp_hands.Hands(  
#     max_num_hands=max_num_hands, # 최대 몇 개의 손을 인식? 
#     min_detection_confidence=0.5, # 0.5로 해두는 게 좋다!  
#     min_tracking_confidence=0.5)  

# # 제스처 인식 모델 
# file = np.genfromtxt('data/gesture_train.csv', delimiter=',') # **각 제스처들의 라벨과 각도가 저장되어 있음, 정확도를 높이고 싶으면 데이터를 추가해보자!** 
# angle = file[:,:-1].astype(np.float32) # 각도
# label = file[:, -1].astype(np.float32) # 라벨
# knn = cv2.ml.KNearest_create() # knn(k-최근접 알고리즘)으로   
# knn.train(angle, cv2.ml.ROW_SAMPLE, label) # 학습! 

# cap = cv2.VideoCapture(0) 

# while cap.isOpened(): # 웹캠에서 한 프레임씩 이미지를 읽어옴
#     ret, img = cap.read()
#     if not ret:
#         continue

#     img = cv2.flip(img, 1)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     result = hands.process(img)

#     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

#     # 각도를 인식하고 제스처를 인식하는 부분 
#     if result.multi_hand_landmarks is not None: # 만약 손을 인식하면 
#         for res in result.multi_hand_landmarks: 
#             joint = np.zeros((21, 3)) # joint == 랜드마크에서 빨간 점, joint는 21개가 있고 x,y,z 좌표니까 21,3
#             for j, lm in enumerate(res.landmark):
#                 joint[j] = [lm.x, lm.y, lm.z] # 각 joint마다 x,y,z 좌표 저장

#             # Compute angles between joints joint마다 각도 계산 
#             # **공식문서 들어가보면 각 joint 번호의 인덱스가 나옴**
#             v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
#             v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
#             v = v2 - v1 # [20,3]관절벡터 
#             # Normalize v
#             v = v / np.linalg.norm(v, axis=1)[:, np.newaxis] # 벡터 정규화(크기 1 벡터) = v / 벡터의 크기

#             # Get angle using arcos of dot product **내적 후 arcos으로 각도를 구해줌** 
#             angle = np.arccos(np.einsum('nt,nt->n',
#                 v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
#                 v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

#             angle = np.degrees(angle) # Convert radian to degree

#             # Inference gesture 학습시킨 제스처 모델에 참조를 한다. 
#             data = np.array([angle], dtype=np.float32)
#             ret, results, neighbours, dist = knn.findNearest(data, 3) # k가 3일 때 값을 구한다! 
#             idx = int(results[0][0]) # 인덱스를 저장! 

#             # Draw gesture result
#             if idx in rps_gesture.keys(): # 만약 인덱스가 가위바위보 중에 있다면 가위바위보 글씨 표시
#                 cv2.putText(img, text=rps_gesture[idx].upper(), org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

#             # Other gestures 모든 제스처를 표시한다면 
#             # cv2.putText(img, text=gesture[idx].upper(), org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

#             mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS) # 손에 랜드마크를 그려줌 

#     cv2.imshow('Game', img)
#     if cv2.waitKey(1) == ord('q'):
#         break

import cv2
import mediapipe as mp 
import math

FRAME_DELAY = 100

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

mp_hands = mp.solutions.hands

mp_fingers = mp_hands.HandLandmark



def run():
    cap = cv2.VideoCapture(0)

    hands = mp_hands.Hands(
        max_num_hands =1,
        model_complexity = 0,
        min_detection_confidence = 0.5,
        min_tracking_confidence = 0.5
    )
    
    while cap.isOpened():
        #모든 비디오 장치의 목록을 받아보고 
        success, image = cap.read()
        if not success:
            print('Ignoring empty camera frame.')
            continue
        
        image = cv2.flip(image,1)
        
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = hands.process(image) #RGB
        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        width, height, _= image.shape
        #exit()
        
    
        #조건문이 참일떄만 실행
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                get_angle(
                    hand_landmarks.landmark[mp_fingers.INDEX_FINGER_MCP],
                    hand_landmarks.landmark[mp_fingers.INDEX_FINGER_TIP],
                    hand_landmarks.landmark[mp_fingers.MIDDLE_FINGER_TIP])
                
                index_finger_tip = hand_landmarks.landmark[mp_fingers.INDEX_FINGER_TIP]
                
                
                cv2.putText(
                    image,
                    text=f'{str(int(index_finger_tip.x * width))}, {str(int(index_finger_tip.y * height))}',
                    org=(100,100),
                    fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(0,0,255),
                    thickness=2
                )
                #result 과정처리 
                mp_drawing.draw_landmarks(
                    image, 
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )
        
        cv2.imshow('MediaPipe Hands', image)
        cv2.waitKey(FRAME_DELAY)
    cap.release()

#new*
def get_angle(ps,p1,p2):
    
    angle1 = math.atan((p1.y -ps.y) / (p1.x -ps.x))
    angle2 = math.atan((p2.y -ps.y) / (p2.x -ps.x))
    
    angle = abs(angle1 - angle2) * 180 / math.pi
    print(f'angle: {angle}')
    return angle
    #exit()
    

    
run()