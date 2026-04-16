import asyncio
import pytest


@pytest.fixture(scope="module")
def event_loop():
    """
    Module-scoped event loop for connectivity tests.

    By default pytest-asyncio creates a new loop per test function. Azure
    Identity's async HTTP transports (httpx / aiohttp) schedule background
    cleanup tasks that need the same loop to still be running when they fire.
    Sharing one loop across the whole module lets those tasks complete cleanly
    and eliminates the 'Event loop is closed' / 'Unclosed client session' noise.
    """
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
