import tensorflow as tf
import numpy as np
from .gesture_detector_interface import HandGestureDetector

class StaticGestureDetector(HandGestureDetector) :
    def __init__(
            self,
            model : str = "static_hand_gesture_model",
            dict : dict = None,
        ) :
        self._interpreter = tf.lite.Interpreter(model_path=f'model/model.tflite/{model}.tflite')
        self._dict = dict

        self._interpreter.allocate_tensors()

        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()

    def detect(self, hand_landmark) :
        input_data = np.array(hand_landmark, dtype=np.float32).reshape(self._input_details[0]['shape'])
        self._interpreter.set_tensor(self._input_details[0]['index'], input_data)
        self._interpreter.invoke()
        output_data = self._interpreter.get_tensor(self._output_details[0]['index'])
        return np.argmax(output_data)
    
    def prediction(self, predicted_class=-1) :
        return self._dict[predicted_class]
    
    def normalize(self, landmarks) :
        coords = np.array([[lm.x, lm.y] for lm in landmarks.landmark], dtype=np.float32)
        coords -= coords[0]
        return coords.ravel()