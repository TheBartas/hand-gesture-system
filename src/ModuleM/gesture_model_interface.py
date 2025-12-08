from abc import ABC, abstractmethod

class HandGestureModel(ABC) :
    def __init__(
            self, 
            camera : int = None,
            model_complexity : int = None,
            max_num_hands : int = None,
            min_detection_confidence : float = None,
            min_tracking_confidence : float = None,
            static_image_mode : bool = False,
            test_size : float = None,
            random_state : int = None,
            optimizer : str = None,
            loss : str = None,
            metrics : list = None,
            epochs : int = None,
            batch_size : int = None,
            instance_name : str = None, # It is good practice to pass the name of the object instance (variable) as the ‘model_name’ argument.
            model_file_name : str = None, # It's a name of model file in .keras and .tflite.
            ) :
        self._camera = camera
        self._model_complexity = model_complexity
        self._max_num_hands = max_num_hands
        self._min_detection_confidence = min_detection_confidence
        self._min_tracking_confidence = min_tracking_confidence
        self._static_image_mode = static_image_mode
        self._test_size = test_size
        self._random_state = random_state
        self._optimizer = optimizer
        self._loss = loss
        self._metrics = metrics
        self._epochs = epochs
        self._batch_size = batch_size
        self._instance_name = instance_name
        self._model_file_name = model_file_name

    @abstractmethod
    def info(self) : pass

    @abstractmethod
    def init_data_files(self) : pass

    @abstractmethod
    def collect_data(self) : pass

    @abstractmethod
    def build_model(self) : pass

    @abstractmethod
    def save(self) : pass

    @abstractmethod
    def convert_to_tflite(self) : pass