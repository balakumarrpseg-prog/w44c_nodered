import asyncio
import sys
from kasa import Discover

IP = "192.168.1.51"

async def main(action):
    plug = await Discover.discover_single(IP)
    try:
        await plug.update()
        if action == "on":
            await plug.turn_on()
        elif action == "off":
            await plug.turn_off()
        await plug.update()
        print("ON" if plug.is_on else "OFF")
    finally:
        await plug.disconnect()

if __name__ == "__main__":
    action = sys.argv[1].lower() if len(sys.argv) > 1 else "status"
    asyncio.run(main(action))