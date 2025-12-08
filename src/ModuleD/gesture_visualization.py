import mediapipe as mp
import cv2
from ModuleD.static_detector import StaticGestureDetector
from ModuleD.dynamic_detector import DynamicGestureDetector

class GestureVisualization:

    @staticmethod
    def continuous_detection(
            staticDetector : StaticGestureDetector=None,
            dynamicDetector : DynamicGestureDetector=None,
            camera=0,
            model_complexity=1,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            static_image_mode=False,
            target_frames=32,
        ) :

        trajectory = []
        seq = []
        frame_counter = 0
        predicted_class = None

        cap = cv2.VideoCapture(camera, cv2.CAP_DSHOW)

        with mp.solutions.hands.Hands(
            model_complexity = model_complexity,
            max_num_hands = max_num_hands,
            min_detection_confidence = min_detection_confidence,
            min_tracking_confidence = min_tracking_confidence,
            static_image_mode=static_image_mode
        ) as hands:
            while cap.isOpened():
                key = cv2.waitKey(1) & 0xFF

                success, image = cap.read()

                image = cv2.flip(image, 1)

                if not success:
                    trajectory.clear()
                    continue

                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                result = hands.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if result.multi_hand_landmarks:
                    frame_counter+=1
                    for hand_landmarks in result.multi_hand_landmarks:
                        mp.solutions.drawing_utils.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp.solutions.hands.HAND_CONNECTIONS,
                            mp.solutions.drawing_utils_styles.get_default_hand_landmarks_style(),
                            mp.solutions.drawing_utils_styles.get_default_hand_connections_style(),)

                        seq.append(dynamicDetector.extract_points(hand_landmarks))

                        if frame_counter == target_frames:
                            predicted_class = dynamicDetector.detect(dynamicDetector.normalize(seq))
                            print(f'Gesture: {dynamicDetector.prediction(predicted_class)}')
                            frame_counter = 0
                            seq.clear()
                            trajectory.clear()

                        else:
                            predicted_class = staticDetector.detect(staticDetector.normalize(hand_landmarks))


                        trajectory.append((hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y))
                        
                        if len(trajectory) > 1:
                            for i in range(1, len(trajectory)):
                                x1, y1 = trajectory[i - 1]
                                x2, y2 = trajectory[i]

                                h, w, _ = image.shape
                                pt1 = (int(x1 * w), int(y1 * h))
                                pt2 = (int(x2 * w), int(y2 * h))

                                cv2.line(image, pt1, pt2, (255, 0, 0), 2)
                        

                    cv2.putText(image, f'Gesture: {staticDetector.prediction(predicted_class)}', (180, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                        cv2.LINE_AA)

                cv2.imshow('V1 Hand Gesture Recognition', image)

                if key == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()


    @staticmethod
    def active_detection(
            staticDetector : StaticGestureDetector=None,
            dynamicDetector : DynamicGestureDetector=None,
            camera=0,
            model_complexity=1,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            static_image_mode=False,
            target_frames=32,
        ) :
        seq = []
        frame_counter = 0
        predicted_class = None
        is_dynamic = False

        cap = cv2.VideoCapture(camera, cv2.CAP_DSHOW)

        with mp.solutions.hands.Hands(
            model_complexity = model_complexity,
            max_num_hands = max_num_hands,
            min_detection_confidence = min_detection_confidence,
            min_tracking_confidence = min_tracking_confidence,
            static_image_mode=static_image_mode
        ) as hands:
            while cap.isOpened():
                key = cv2.waitKey(1) & 0xFF

                success, image = cap.read()

                image = cv2.flip(image, 1)

                if not success:
                    continue

                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                result = hands.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if result.multi_hand_landmarks:
                    for hand_landmarks in result.multi_hand_landmarks:
                        mp.solutions.drawing_utils.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp.solutions.hands.HAND_CONNECTIONS,
                            mp.solutions.drawing_utils_styles.get_default_hand_landmarks_style(),
                            mp.solutions.drawing_utils_styles.get_default_hand_connections_style(),
                        )

                        if is_dynamic:
                            seq.append(dynamicDetector.extract_points(hand_landmarks))
                            frame_counter+=1

                            if frame_counter == target_frames:
                                predicted_class = dynamicDetector.detect(dynamicDetector.normalize(seq))
                                print(f'Gesture: {dynamicDetector.prediction(predicted_class)}')

                                frame_counter = 0
                                seq.clear()

                                if predicted_class > 0:
                                    is_dynamic = False

                                continue

                        else:
                            predicted_class = staticDetector.detect(staticDetector.normalize(hand_landmarks))
                            print(predicted_class)
                            if predicted_class == 1: 
                                is_dynamic = True
                                seq.clear()
                                frame_counter = 0

                    cv2.putText(image, f'Gesture: {staticDetector.prediction(predicted_class)}', (180, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                        cv2.LINE_AA)
                    
                cv2.imshow('V1 Hand Gesture Recognition', image)

                if key == ord('q'):
                    break




# if __name__=='__main__':
#     detector1 = StaticGestureDetector(
#         model='static_model',
#         dict = {
#             0:'None',
#             1:'OK',
#             2:'one_finger',
#             3:'thumb_up',
#             4:'two_fingers',
#         }
#     )
#     detector2 = DynamicGestureDetector(
#         model='dynamic_model',
#         dict = {
#             0:'None',
#             1:'one_finger_circle',
#             2:'one_finger_down',
#             3:'one_finger_left'
#         }
#     )

#     GestureVisualization.continuous_detection(
#         staticDetector=detector1, 
#         dynamicDetector=detector2,
#         # camera=CAMERA,
#         # model_complexity=MODEL_COMPLEXITY,
#         # max_num_hands=NUM_HANDS,
#         # min_detection_confidence=MIN_DETECTION_CONFIDENCE,
#         # min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
#         target_frames=40
#     )

#     GestureVisualization.active_detection(
#         staticDetector=detector1, 
#         dynamicDetector=detector2,
#         target_frames=40
#     )