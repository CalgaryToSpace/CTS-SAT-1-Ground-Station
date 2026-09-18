# pyright: standard
# Starlette's `TestClient` is typed through an httpx shim pyright can't
# follow in strict mode -- every `client.get(...)` comes back as an
# unknown type, which says nothing about this module's own types.

import pytest
from cts1_mo_tools.cts1_processing_pipeline.web_ui.picam_image_route import (
    MAX_CACHED_IMAGES,
    picam_image_url,
    register_picam_image_route,
)
from fastapi import FastAPI
from fastapi.testclient import TestClient

JPG_BYTES = b"\xff\xd8\xff\xe0\x00\x10JFIF pretend this is a PiCAM thumbnail"


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    register_picam_image_route(app)
    return TestClient(app)


def test_serves_cached_image_under_its_filename(client: TestClient) -> None:
    url = picam_image_url(JPG_BYTES, "log_b51a.jpg")

    # The filename is the URL's last segment: that's what a browser offers
    # as the name in "Save image as...", which is the point of the route.
    assert url.endswith("/log_b51a.jpg")

    response = client.get(url)

    assert response.status_code == 200
    assert response.content == JPG_BYTES
    assert response.headers["content-type"] == "image/jpeg"
    # `inline` so the preview still renders in the page instead of
    # downloading, with the name alongside it for a right-click save.
    assert response.headers["content-disposition"] == (
        "inline; filename*=UTF-8''log_b51a.jpg"
    )


def test_awkward_filename_survives_the_round_trip(client: TestClient) -> None:
    """A filename comes from a downlinked header, so it can hold anything --
    including characters that would break a raw header value or URL path.
    """
    url = picam_image_url(JPG_BYTES, 'weird "name" é.jpg')

    response = client.get(url)

    assert response.status_code == 200
    assert response.content == JPG_BYTES


def test_same_bytes_reuse_one_cache_entry(client: TestClient) -> None:
    """Content addressed: re-rendering a page with the same image shouldn't
    cost another slot in a cache bounded at `MAX_CACHED_IMAGES`.
    """
    first = picam_image_url(JPG_BYTES, "log_b51a.jpg")
    second = picam_image_url(JPG_BYTES, "log_b51a.jpg")

    assert first == second
    assert client.get(first).content == JPG_BYTES


def test_unknown_token_is_404(client: TestClient) -> None:
    assert client.get("/picam-image/0123456789abcdef/log_b51a.jpg").status_code == 404


def test_cache_is_bounded_evicting_oldest_first(client: TestClient) -> None:
    oldest = picam_image_url(b"the first image", "first.jpg")
    newer = [
        picam_image_url(f"image {i}".encode(), "x.jpg")
        for i in range(MAX_CACHED_IMAGES)
    ]

    # Pushed out by the images that came after it -- a stale preview 404s
    # rather than the server holding every image it has ever rendered.
    assert client.get(oldest).status_code == 404
    assert client.get(newer[-1]).status_code == 200
