"""Redis side of the publish step."""
import shutil
import subprocess
import time

import pytest

from publish import clear_cache


@pytest.mark.skipif(shutil.which("redis-server") is None, reason="redis-server not installed")
def test_clear_cache_removes_only_recommendation_keys():
    import redis
    server = subprocess.Popen(["redis-server", "--port", "6397", "--save", ""],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        client = redis.Redis(port=6397, decode_responses=True)
        for _ in range(50):
            try:
                client.ping()
                break
            except redis.ConnectionError:
                time.sleep(0.1)
        client.set("rec:v1:1:10", "[1]")
        client.set("rec:v1:2:10", "[2]")
        client.set("something:else", "keep")

        assert clear_cache("localhost:6397", "rec:v1") == 2
        assert client.keys("*") == ["something:else"]
    finally:
        server.terminate()


def test_clear_cache_survives_an_unreachable_redis():
    assert clear_cache("localhost:6399", "rec:v1") is None
