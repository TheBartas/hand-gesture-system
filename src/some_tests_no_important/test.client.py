import asyncio
import websockets
import json  

async def client():
    url = "ws://127.0.0.1:8765"
    prev = 'None'
    cnt = 0
    try:
        async with websockets.connect(url) as ws:
            print("Connected to server")
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
                    
                    if cnt != 10:
                        continue

                    print(f"gesture: {gesture}")
                    
                    continue
                if type == "dynamic":
                    print(f"gesture: {gesture}")
                    continue

    except (ConnectionRefusedError, OSError):
        print("Server is not running. Please start the server first.")
    except websockets.ConnectionClosed:
        print("Connection closed")

asyncio.run(client())
