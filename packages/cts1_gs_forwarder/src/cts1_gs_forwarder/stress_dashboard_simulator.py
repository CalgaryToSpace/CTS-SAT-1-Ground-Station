import asyncio
import json
import random
import time
import uuid
import websockets

class DashboardSimulator:
    def __init__(self, dashboard_id: str, api_key: str, url: str):
        self.dashboard_id = dashboard_id
        self.api_key = api_key
        self.url = url
        self.ws = None

    async def connect(self):
        print(f"[{self.dashboard_id}] connecting...")
        self.ws = await websockets.connect(self.url)

        # Send auth
        await self.ws.send(json.dumps({
            "type": "auth",
            "dashboard_id": self.dashboard_id,
            "api_key": self.api_key,
            "ts": time.time()
        }))

        response = await self.ws.recv()
        print(f"[{self.dashboard_id}] auth response:", response)

    async def send_ping(self):
        await self.ws.send(json.dumps({
            "type": "ping",
            "nonce": str(uuid.uuid4()),
            "ts": time.time()
        }))

    async def send_telemetry(self):
        await self.ws.send(json.dumps({
            "type": "telemetry",
            "payload": {
                "metric": "cpu_usage",
                "value": round(random.uniform(10, 90), 2),
                "unit": "%",
                "tags": {
                    "host": self.dashboard_id
                }
            },
            "ts": time.time()
        }))

    async def send_event(self):
        await self.ws.send(json.dumps({
            "type": "event",
            "payload": {
                "name": "threshold_breach",
                "severity": random.choice(["info", "warning", "error"]),
                "data": {
                    "value": random.randint(80, 100)
                }
            },
            "ts": time.time()
        }))

    async def receiver(self):
        try:
            async for message in self.ws:
                print(f"[{self.dashboard_id}] received:", message)
        except websockets.ConnectionClosed:
            print(f"[{self.dashboard_id}] connection closed")

    async def run(self):
        await self.connect()

        receiver_task = asyncio.create_task(self.receiver())

        try:
            while True:
                # Randomly pick an action
                action = random.choice([
                    self.send_ping,
                    self.send_telemetry,
                    self.send_event
                ])

                await action()

                await asyncio.sleep(0.1)

        except Exception as e:
            print(f"[{self.dashboard_id}] error:", e)

        finally:
            await self.ws.close()
            receiver_task.cancel()


async def main():
    url = "ws://localhost:8000/ws"

    dashboards = [
        ("dashboard-alpha", "super-secret-alpha")
        for _ in range(100)
    ]

    simulators = [
        DashboardSimulator(d_id, api_key, url)
        for d_id, api_key in dashboards
    ]

    await asyncio.gather(*(sim.run() for sim in simulators))


if __name__ == "__main__":
    asyncio.run(main())