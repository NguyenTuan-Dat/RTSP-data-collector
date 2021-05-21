import asyncio

from aiohttp import ClientSession

from eufy_security import async_login


async def main() -> None:
    """Create the aiohttp session and run the example."""
    async with ClientSession() as websession:
        # Create an API client:
        api = await async_login("binh.duongsy@gmail.com", "!K1n90fC2s", websession)

        # Loop through the cameras associated with the account:
        for camera in api.cameras.values():
            print("------------------")
            print("Camera Name: %s", camera.name)
            print("Serial Number: %s", camera.serial)
            print("Station Serial Number: %s", camera.station_serial)
            print("Last Camera Image URL: %s", camera.last_camera_image_url)

            print("Starting RTSP Stream")
            stream_url = await camera.async_start_stream()
            print("Stream URL: %s", stream_url)

            print("Stopping RTSP Stream")
            stream_url = await camera.async_stop_stream()


asyncio.get_event_loop().run_until_complete(main())