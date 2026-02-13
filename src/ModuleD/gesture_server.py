import asyncio
import websockets
import cv2
import json
import threading as th
import mediapipe as mp
from ModuleD import StaticGestureDetector, DynamicGestureDetector


from config import (
    CAMERA, 
    MODEL_COMPLEXITY, 
    NUM_HANDS, 
    MIN_DETECTION_CONFIDENCE, 
    MIN_TRACKING_CONFIDENCE,
    TARGET_FRAMES
)

class GestureServer:
    def __init__(
            self,
            staticDetector : StaticGestureDetector = None,
            dynamicDetector : DynamicGestureDetector = None,
            host='127.0.0.1',
            port=8765,
            camera=0,
            model_complexity=1,
            max_num_hands = 1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            static_image_mode=False,
            target_frames=40,
            mode='continuous'
        ) :
        self._staticDetector=staticDetector
        self._dynamicDetector=dynamicDetector
        self._host=host
        self._port=port
        self._camera=camera
        self._model_complexity=model_complexity
        self._max_num_hands=max_num_hands
        self._min_detection_confidence=min_detection_confidence
        self._min_tracking_confidence=min_tracking_confidence
        self._static_image_mode=static_image_mode
        self._TARGET_FRAMES=target_frames
        self._mode=mode
        
        self._gesture = 'None'
        self._type = 'static'

    async def run(self) :
        print(f'Server running on ws://127.0.0.1:{self._port}')
        th.Thread(target=self.camera_loop, daemon=True).start()
        server = await websockets.serve(self.handle_client, "127.0.0.1", self._port)
        await server.wait_closed()

    async def handle_client(self, websocket) :
        try:
            while True:
                await websocket.send(json.dumps({'gesture': self._gesture, "type": self._type}))
                await asyncio.sleep(0.033)
        except websockets.ConnectionClosed:
            print('[info] Connection closed by client.')

    def camera_loop(self) :
        if self._mode == 'continuous':
            self.continuous_detection()
        elif self._mode == 'active':
            self.active_detection()
        else :
            return ValueError("[Error] Unknown detection mode.")

    def continuous_detection(self) :
        self.d_gesture = [] # for dynamic hand landmark sequence
        
        seq = []
        frame_counter = 0
        predicted_class = 'None'

        cap = cv2.VideoCapture(self._camera, cv2.CAP_DSHOW)
        with mp.solutions.hands.Hands(
            model_complexity = self._model_complexity,
            max_num_hands = self._max_num_hands,
            min_detection_confidence = self._min_detection_confidence,
            min_tracking_confidence = self._min_tracking_confidence,
            static_image_mode=self._static_image_mode
        ) as hands:
            print("[info] Camera run.")
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
                            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                            mp.solutions.drawing_styles.get_default_hand_connections_style(),
                        )
                        frame_counter+=1
                        
                        seq.append(self._dynamicDetector.extract_points(hand_landmarks))
                        
                        if frame_counter == self._TARGET_FRAMES:
                            pred_dyn = self._dynamicDetector.detect(self._dynamicDetector.normalize(seq))
                            frame_counter = 0
                            seq.clear()
                            if pred_dyn > 0:
                                self._gesture = self._dynamicDetector.prediction(pred_dyn)
                                self._type = 'dynamic'
                            else:
                                self._gesture = 'None'
                                self._type = 'dynamic'
                        else:
                            predicted_class = self._staticDetector.detect(self._staticDetector.normalize(hand_landmarks))
                            self._gesture = self._staticDetector.prediction(predicted_class)
                            self._type = 'static'
                else:
                    frame_counter = 0
                    seq.clear()
                    self._gesture = 'None'
                    self._type = 'static'

                cv2.imshow('MediaPipe Hands', image)

                if key == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

    def active_detection(self) :
        self.d_gesture = [] # for dynamic hand landmark sequence
        
        seq = []
        frame_counter = 0
        predicted_class = 'None'
        is_dynamic = False

        cap = cv2.VideoCapture(self._camera, cv2.CAP_DSHOW)
        with mp.solutions.hands.Hands(
            model_complexity = self._model_complexity,
            max_num_hands = self._max_num_hands,
            min_detection_confidence = self._min_detection_confidence,
            min_tracking_confidence = self._min_tracking_confidence,
            static_image_mode=self._static_image_mode
        ) as hands:
            print("[info] Camera run.")
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
                            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                            mp.solutions.drawing_styles.get_default_hand_connections_style(),
                        )
                        
                        if is_dynamic:
                            seq.append(self._dynamicDetector.extract_points(hand_landmarks))
                            frame_counter+=1

                            if frame_counter == TARGET_FRAMES:
                                predicted_class = self._dynamicDetector.detect(self._dynamicDetector.normalize(seq))

                                frame_counter = 0
                                seq.clear()

                                if predicted_class > 0:
                                    is_dynamic = False
                                    self._gesture = self._dynamicDetector.prediction(predicted_class)
                                    self._type = 'dynamic'

                                continue

                        else:
                            predicted_class = self._staticDetector.detect(self._staticDetector.normalize(hand_landmarks))
                            if predicted_class == 1: 
                                is_dynamic = True
                                seq.clear()
                                frame_counter = 0
                            self._gesture = self._staticDetector.prediction(predicted_class)
                            self._type = 'static'
                else:
                    self._gesture = 'None'
                    self._type = 'static'

                cv2.imshow('MediaPipe Hands', image)

                if key == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    detector1 = StaticGestureDetector(
        model='static_model',
        dict = {
            0:'None',
            1:'OK',
            2:'one_finger',
            3:'thumb_up',
            4:'two_fingers',
        }
    )
    detector2 = DynamicGestureDetector(
        model='dynamic_model',
        dict = {
            0:'None',
            1:'one_finger_circle',
            2:'one_finger_down',
            3:'one_finger_left'
        }
    )

    gesture_server = GestureServer(
        staticDetector=detector1,
        dynamicDetector=detector2,
        camera=CAMERA,
        model_complexity = MODEL_COMPLEXITY,
        max_num_hands = NUM_HANDS,
        min_detection_confidence = MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence = MIN_TRACKING_CONFIDENCE,
        target_frames=TARGET_FRAMES,
        mode='active',

    )
    try:
        asyncio.run(gesture_server.run())
    except KeyboardInterrupt:
        print('[info] Server closed by user.')