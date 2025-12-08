import json
import csv
import cv2
import pandas as pd
import mediapipe as mp
import numpy as np
import tensorflow as tf
from ModuleM.gesture_model_interface import HandGestureModel
from keras.models import Sequential
from keras.layers import Dropout, Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from ModuleM.paths import STATIC_PATH, MODEL_KERAS_PATH, MODEL_TFLITE_PATH, INFO_PATH, MODEL_JSON_PATH

class StaticGestureModel(HandGestureModel) :

    def __init__(
        self, 
        camera=0,
        model_complexity=1,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        static_image_mode=False,
        test_size=0.2,
        random_state=42,
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
        epochs=20,
        batch_size=34,
        instance_name='static_default',
        model_file_name='static_hand_gesture_model_default',
        ) : 
        super().__init__(
            camera=camera,
            model_complexity=model_complexity,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            static_image_mode=static_image_mode,
            test_size=test_size,
            random_state=random_state,
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            epochs=epochs,
            batch_size=batch_size,
            instance_name=instance_name, 
            model_file_name=model_file_name,
            )

    def __str__(self) : 
        return f"{self._instance_name} | {type(self).__name__}"

    def info(self) : 
        print(f"\n######## Model<{self._instance_name}> Info ########")

        for attr, value in self.__dict__.items():
            print(f"| {attr.lstrip('_').replace('_', ' ').title()}: {value}")
    
        print("###---------------------------------###\n\n")

    def init_data_files(self) : 
        with open(INFO_PATH, 'r') as file:
            data = json.load(file)

        names = [gesture['name'] for gesture in data if gesture['type'] == 'static']

        for name in names:
            path = STATIC_PATH / f'{name}_data.csv'
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()

                header = ['gesture_name']
                num_points = 21 # it's number o MediaPipe's points

                for point in range(num_points):
                    header.append(f"x{point}")
                    header.append(f"y{point}")

                with open(path, 'a', newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(header)
    
    def collect_data(self) : 
        with open(INFO_PATH, 'r') as file:
            data = json.load(file)

        data = pd.DataFrame(data)

        static_gestures = data[data['type'] == 'static']
        print(static_gestures)

        gesture_id = input("[Info] Enter ID (left column from dataframe) of the gesture you want to record: ").strip()
        gesture_name = data['name'].iloc[int(gesture_id)]
        print(f'Gesture: <{gesture_name}>')

        path = STATIC_PATH / f'{gesture_name}_data.csv'
        hand_data_to_file = []
        coords = []

        if not path.exists():
            print(f'[Error] File for <{gesture_name}> does not exists. Try <init_data_files> before.')
            return -1

        cap = cv2.VideoCapture(self._camera, cv2.CAP_DSHOW)
        with mp.solutions.hands.Hands(
            model_complexity = self._model_complexity,
            max_num_hands = self._max_num_hands,
            min_detection_confidence = self._min_detection_confidence,
            min_tracking_confidence = self._min_tracking_confidence,
            static_image_mode=self._static_image_mode    
        ) as hands:

            while cap.isOpened() :
                key = cv2.waitKey(1) & 0xFF

                success, image = cap.read()

                if not success:
                    continue

                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.multi_hand_landmarks:
                    for hand_landmark in results.multi_hand_landmarks: # hand_landmark is a list of 21 points from one hand
                        mp.solutions.drawing_utils.draw_landmarks (
                            image,
                            hand_landmark,
                            mp.solutions.hands.HAND_CONNECTIONS,
                            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                            mp.solutions.drawing_styles.get_default_hand_connections_style()
                        )
                        
                        coords = self.__normalize(hand_landmark)

                if key == ord("t"):
                    hand_data_to_file.append([gesture_name] + coords.tolist())
                    print(f"Data saved! {len(coords)}")
                    
                cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))

                if key == ord('q'):
                    break
        cap.release()
        cv2.destroyAllWindows()

        with open(path, "a", newline="") as data:
            writer = csv.writer(data)
            writer.writerows(hand_data_to_file)
                   

    def build_model(self) : 
        files = STATIC_PATH.rglob('*.csv')

        X = []
        y = []

        for file in files:
            data_to_learn = pd.read_csv(file)
            X.append(data_to_learn.drop('gesture_name', axis=1).values)
            y.extend(data_to_learn['gesture_name'].values) # it must be extend, because LabelEncoder requiers 1d numpy array

        X = np.vstack(X)
        y = np.array(y)

        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, 
            y, 
            test_size=self._test_size,
            random_state=self._random_state)

        self._model = Sequential([
            Input(shape=(42,)),
            Dense(128, activation="relu"),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(len(np.unique(y)), activation="softmax")
        ])

        self._model.compile(
                optimizer=self._optimizer,
                loss=self._loss,
                metrics=self._metrics
            )

        self._model.fit(X_train, y_train, epochs=self._epochs, batch_size=self._batch_size, validation_data=(X_test, y_test))

    def save(self) : 
        path = MODEL_KERAS_PATH / f'{self._model_file_name}.keras'
        self._model.save(path)
        print(f'[info] Model saved at: {path}')

    def convert_to_tflite(self) : 
        self._model_tflite = tf.lite.TFLiteConverter.from_keras_model(self._model).convert()
        path = MODEL_TFLITE_PATH / f'{self._model_file_name}.tflite'

        with open(path, 'wb') as file:
            file.write(self._model_tflite)
        
        print(f'[info] Model converted successfully at: {path}')

    def to_json(self) :
        data = {
            "class": "static",
            "data": {
                "camera": self._camera,
                "model_complexity": self._model_complexity,
                "max_num_hands": self._max_num_hands,
                "min_detection_confidence": self._min_detection_confidence,
                "min_tracking_confidence": self._min_tracking_confidence,
                "static_image_mode": self._static_image_mode,
                "test_size": self._test_size,
                "random_state": self._random_state,
                "optimizer": self._optimizer,
                "loss": self._loss,
                "metrics": self._metrics,
                "epochs": self._epochs,
                "batch_size": self._batch_size,
                "instance_name": self._instance_name,
                "model_file_name": self._model_file_name,
            }
        }

        with open(MODEL_JSON_PATH / f'{self._model_file_name}.json', "w") as file:
            json.dump(data, file, indent=4)

    @classmethod
    def from_json(cls, file_name=None, model=None):

        if file_name is not None:
            with open(MODEL_JSON_PATH / f"{file_name}.json") as f:
                model = json.load(f)

            if model["class"] != 'static' :
                print("[Error] Wrong model class!")
                return False

        if model is None:
            raise ValueError("[Error] Provide either <file_name> or <model> JSON dict")
 
        data = model["data"]

        return cls(
            camera=data["camera"],
            model_complexity=data["model_complexity"],
            max_num_hands=data["max_num_hands"],
            min_detection_confidence=data["min_detection_confidence"],
            min_tracking_confidence=data["min_tracking_confidence"],
            static_image_mode=data["static_image_mode"],
            test_size=data["test_size"],
            random_state=data["random_state"],
            optimizer=data["optimizer"],
            loss=data["loss"],
            metrics=data["metrics"],
            epochs=data["epochs"],
            batch_size=data["batch_size"],
            instance_name=data["instance_name"],
            model_file_name=data["model_file_name"],
        )
    
    def __normalize(self, landmarks) :
        coords = np.array([[lm.x, lm.y] for lm in landmarks.landmark])
        coords -= coords[0]
        return coords.ravel()
    

    
