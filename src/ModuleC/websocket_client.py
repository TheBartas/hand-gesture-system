import websockets
import json
from ModuleC import GestureHandler

class WebsocketClient:
    def __init__(
            self,
            handler : GestureHandler = None,
            ws_host : str = 'ws://127.0.0.1',
            ws_port : str = '8765',
        ):
        self._handler = handler
        self.url = ws_host + ':' + ws_port

    async def run(self) :
        prev = 'None'
        cnt = 0
        try:
            async with websockets.connect(self.url) as ws:
                print(f"[info] Connected to gesture server running at {self.url}")
                while True:
                    msg = await ws.recv()
                    data = json.loads(msg)

                    gesture = data["gesture"]
                    type = data["type"]

                    if type == "static":
                        if prev != gesture:
                            prev = gesture
                            cnt=0
                        elif prev == gesture:
                            cnt+=1
                        if cnt == 15:
                            await self._handler.handle(gesture)
                        continue
                    if type == "dynamic":
                        await self._handler.handle(gesture)
                        continue

        except (ConnectionRefusedError, OSError):
            print("[info] Server is not running. Please start the server first.")
        except websockets.ConnectionClosed:
            print("[info] Connection closed")
        except KeyboardInterrupt:
            print("[info] Keyboard interrupt detected.")