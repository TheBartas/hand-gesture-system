import asyncio
import os
from ModuleC import PTZCameraController, VirtualController, GestureHandler, WebsocketClient
from dotenv import load_dotenv


load_dotenv()
CAMERA_IP=os.getenv('CAMERA_IP')
USERNAME=os.getenv('USERNAME')
PASSWORD=os.getenv('PASSWORD')


async def main() :
    # camera = PTZCameraController(
    #     camera_ip=CAMERA_IP,
    #     username=USERNAME,
    #     password=PASSWORD,
    # )

    controller = VirtualController(
        ws_port='8080',
        ws_path='ptz',
    )
    await controller.connect()

    handler = GestureHandler(
        controller=controller,
        pan_left=-1,
        pan_right=1,
        tilt_up=1,
        tilt_down=-1,
        zoom_in=1,
        zoom_out=-1,
    )
    ws = WebsocketClient(handler)

    await ws.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("[info] Controller closed.")


