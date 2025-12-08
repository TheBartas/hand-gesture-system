from abc import ABC, abstractmethod

class HandGestureDetector(ABC):

    @abstractmethod
    def detect(self, landmarks) :
        pass

    @abstractmethod
    def prediction(self, predicted_class) :
        pass

    @abstractmethod
    def normalize(self, landmarks) :
        pass
    

    
    