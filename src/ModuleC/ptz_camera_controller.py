import base64
import aiohttp
from ModuleC import Controller

class PTZCameraController(Controller):
    def __init__(
        self,
        camera_ip : str = '127.0.0.1',
        username : str = 'username',
        password : str = 'password',
        pan : int = 0,
        tilt : int = 0,
        zoom : int = 0,
    ) : 
        self._camera_ip = camera_ip
        self._username = username
        self._password = password
        self._pan = pan
        self._tilt = tilt
        self._zoom = zoom

        user_pass=f'{username}:{password}'.encode('ascii')
        self._auth_header = {"Authorization": "Basic " + base64.b64encode(user_pass).decode("ascii")}
        self._url = f'http://{camera_ip}/axis-cgi/com/ptz.cgi?'
        # self._url = 'http://127.0.0.1:8080?' # for fake server, only for tests


    async def _send_command(self, url) :
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, headers=self._auth_header) as response:
                    if response.status == 200:
                        print("[info] PTZ command sent successfully.")
                    else:
                        html = await response.text()
                        print(f"[Error] {response.status} - {html}")
            except Exception as ex:
                print(f"[Error] {ex}")

    async def pan(self, pan) :
        self._pan+=pan
        url = self._url + f'pan={self._pan}'
        await self._send_command(url)

    async def tilt(self, tilt) :
        self._tilt+=tilt
        url = self._url + f'tilt={self._tilt}'
        await self._send_command(url)

    async def zoom(self, zoom) :
        self._zoom+=zoom
        url = self._url + f'tilt={self._zoom}'
        await self._send_command(url)
