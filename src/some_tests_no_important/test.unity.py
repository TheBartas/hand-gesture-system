import asyncio
import websockets

connected_clients = set()

async def ptz_handler(websocket):
    connected_clients.add(websocket)
    print(f"New client connected: {websocket.remote_address}")

    try:
        while True:
            msg = await websocket.recv()
            print(f"Received: {msg}")
    except websockets.ConnectionClosed:
        print(f"[info] Client disconnected: {websocket.remote_address}")
    finally:
        connected_clients.remove(websocket)


async def main():
    server = await websockets.serve(ptz_handler, "localhost", 8080)
    print("WebSocket server running at ws://localhost:8080")
    await server.wait_closed()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"[info] Keyboard interrupt detected.")
