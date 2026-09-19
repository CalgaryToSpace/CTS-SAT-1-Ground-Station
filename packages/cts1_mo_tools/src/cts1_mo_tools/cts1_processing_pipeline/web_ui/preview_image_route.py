"""Serving of previewable images over a real URL, so that a right-click >
"Save image as..." on the File Reassembler's inline preview opens pre-filled
with the reassembled file's own name.

A browser takes that suggested name from the image's URL (and its
`Content-Disposition`), neither of which a `data:` URI has -- an inline
base64 image saves as "download.jpg" in Chromium no matter what parameters
the URI carries. So the preview points at `GET {PREVIEW_IMAGE_PATH}/{token}/
{filename}` instead, ending in the real filename, and this module holds the
bytes in between the page render and the browser's fetch.

The image only exists in memory (a downlinked .jpg/.bmp is reassembled from
packets on the fly, a PiCAM image decoded from them, and neither is written
to `output/`), so that hand-off is a small bounded cache rather than a file
on disk: content addressed, so re-rendering the same image reuses its entry,
and capped at `MAX_CACHED_IMAGES` so a long-lived server walking through
many reassemblies can't accumulate them without limit.
"""

__all__ = [
    "MAX_CACHED_IMAGES",
    "PREVIEW_IMAGE_PATH",
    "preview_image_url",
    "register_preview_image_route",
]

import hashlib
from collections import OrderedDict
from typing import NamedTuple
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Response

PREVIEW_IMAGE_PATH = "/preview-image"

# A downlinked image is small -- a PiCAM thumbnail is tens of KB, and a
# bulk download big enough to be worth previewing is still a few MB at
# most -- so this caps the hold-over at a manageable size, while being far
# more than the handful of images any one page render puts on screen.
MAX_CACHED_IMAGES = 32

# Long enough to identify an image without the URL turning into a wall of
# hex; a collision only means one preview shows another's (identical-length)
# bytes, and these are process-local and short-lived.
_TOKEN_LEN = 16


class _CachedImage(NamedTuple):
    data: bytes
    media_type: str


_cached_images: OrderedDict[str, _CachedImage] = OrderedDict()


def preview_image_url(data: bytes, media_type: str, filename: str) -> str:
    """Cache `data` and return the URL serving it back as `media_type`,
    ending in `filename` so that's what a browser offers to save it as.
    """
    token = hashlib.sha256(data).hexdigest()[:_TOKEN_LEN]
    _cached_images[token] = _CachedImage(data, media_type)
    _cached_images.move_to_end(token)
    while len(_cached_images) > MAX_CACHED_IMAGES:
        _cached_images.popitem(last=False)
    return f"{PREVIEW_IMAGE_PATH}/{token}/{quote(filename, safe='')}"


def register_preview_image_route(app: FastAPI) -> None:
    """Register `GET {PREVIEW_IMAGE_PATH}/{token}/{filename}` on `app`, serving
    images cached by `preview_image_url`.
    """

    def preview_image(token: str, filename: str) -> Response:
        cached = _cached_images.get(token)
        if cached is None:
            # Evicted, or served by a previous run of the server: the page
            # that built this URL is the only thing that can produce the
            # bytes again, so re-running the reassembly is the only fix.
            raise HTTPException(
                status_code=404,
                detail="This image is no longer cached -- reassemble the file again.",
            )
        _cached_images.move_to_end(token)

        return Response(
            cached.data,
            media_type=cached.media_type,
            headers={
                # `inline` so it still renders in the page rather than
                # downloading; the filename rides along for "Save image
                # as...". RFC 5987's `filename*` form keeps non-ASCII names
                # (and quotes) from breaking a latin-1 header value.
                "Content-Disposition": (
                    f"inline; filename*=UTF-8''{quote(filename, safe='')}"
                ),
                # Content addressed: a given token's bytes never change.
                "Cache-Control": "private, max-age=3600, immutable",
            },
        )

    # `add_api_route` rather than `@app.get(...)` for the same reason as
    # `export_raw.register_raw_export_route` -- see the note there.
    app.add_api_route(
        PREVIEW_IMAGE_PATH + "/{token}/{filename}", preview_image, methods=["GET"]
    )
