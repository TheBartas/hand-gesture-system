import websockets
import json
from ModuleC import Controller


class VirtualController(Controller):
    def __init__(
        self,
        ws_host : str = '127.0.0.1',
        ws_port : str = '8675',
        ws_path : str = 'virtual/camera',
    ) : 
        self._url = f'ws://{ws_host}:{ws_port}/{ws_path}'
        self.ws = None

    async def connect(self):
        try:
            self.ws = await websockets.connect(self._url)
            print(f"[info] Connected to server running at {self._url}")
        except (OSError, websockets.exceptions.WebSocketException) as e:
            print(f"[error] From Controller: Cannot connect: {e}")

    async def close(self):
        if self.ws:
            await self.ws.close()

    async def _send_command(self, command : dict) :
        await self.ws.send(json.dumps(command)) 
        

    async def pan(self, value) :
        await self._send_command({"type":"pan","value":value})

    async def tilt(self, value) :
        await self._send_command({"type":"tilt","value":value})

    async def zoom(self, value) :
        await self._send_command({"type":"zoom","value":value})