from abc import ABC, abstractmethod

class Controller(ABC):

    @abstractmethod
    async def pan(self, pan) : pass

    @abstractmethod
    async def tilt(self, tilt) : pass

    @abstractmethod
    async def zoom(self, zoom) : pass

