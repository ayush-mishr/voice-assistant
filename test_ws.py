"""Quick test: connect to local backend, start a session, and check for responses."""
import asyncio
import websockets
import json

async def test():
    uri = "ws://localhost:8000/ws"
    try:
        async with websockets.connect(uri) as ws:
            print("1. WebSocket connected OK")
            
            # Send session_start
            await ws.send(json.dumps({"type": "session_start"}))
            print("2. Sent session_start, waiting for response...")
            
            # Wait for responses (should get state: listening)
            for i in range(5):
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=10)
                    if isinstance(msg, bytes):
                        print(f"3. Got binary audio ({len(msg)} bytes)")
                    else:
                        data = json.loads(msg)
                        print(f"3. Got JSON: {data}")
                except asyncio.TimeoutError:
                    print(f"   Timeout after 10s waiting for message #{i+1}")
                    break
            
            print("4. Test complete")
    except Exception as e:
        print(f"ERROR: {e}")

asyncio.run(test())
