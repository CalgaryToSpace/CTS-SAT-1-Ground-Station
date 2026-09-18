"""Serving of detected PiCAM images over a real URL, so that a right-click
> "Save image as..." on the File Reassembler's inline preview opens
pre-filled with the reassembled file's own name.

A browser takes that suggested name from the image's URL (and its
`Content-Disposition`), neither of which a `data:` URI has -- an inline
base64 image saves as "download.jpg" in Chromium no matter what parameters
the URI carries. So the preview points at `GET {PICAM_IMAGE_PATH}/{token}/
{filename}` instead, ending in the real filename, and this module holds the
bytes in between the page render and the browser's fetch.

The JPG only exists in memory (it's decoded from packets on the fly by
`file_reassembly.detect_picam_image`, never written to `output/`), so that
hand-off is a small bounded cache rather than a file on disk: content
addressed, so re-rendering the same image reuses its entry, and capped at
`MAX_CACHED_IMAGES` so a long-lived server walking through many
reassemblies can't accumulate them without limit.
"""

__all__ = [
    "MAX_CACHED_IMAGES",
    "PICAM_IMAGE_PATH",
    "picam_image_url",
    "register_picam_image_route",
]

import hashlib
from collections import OrderedDict
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Response

PICAM_IMAGE_PATH = "/picam-image"

# Each PiCAM image is a downlinked thumbnail -- tens of KB -- so this caps
# the hold-over at a few MB, while still being far more than the handful of
# images any one page render puts on screen at once.
MAX_CACHED_IMAGES = 32

# Long enough to identify an image without the URL turning into a wall of
# hex; a collision only means one preview shows another's (identical-length)
# bytes, and these are process-local and short-lived.
_TOKEN_LEN = 16

_cached_images: OrderedDict[str, bytes] = OrderedDict()


def picam_image_url(jpg_bytes: bytes, filename: str) -> str:
    """Cache `jpg_bytes` and return the URL serving them back, ending in
    `filename` so that's what a browser offers to save them as.
    """
    token = hashlib.sha256(jpg_bytes).hexdigest()[:_TOKEN_LEN]
    _cached_images[token] = jpg_bytes
    _cached_images.move_to_end(token)
    while len(_cached_images) > MAX_CACHED_IMAGES:
        _cached_images.popitem(last=False)
    return f"{PICAM_IMAGE_PATH}/{token}/{quote(filename, safe='')}"


def register_picam_image_route(app: FastAPI) -> None:
    """Register `GET {PICAM_IMAGE_PATH}/{token}/{filename}` on `app`, serving
    images cached by `picam_image_url`.
    """

    def picam_image(token: str, filename: str) -> Response:
        jpg_bytes = _cached_images.get(token)
        if jpg_bytes is None:
            # Evicted, or served by a previous run of the server: the page
            # that built this URL is the only thing that can produce the
            # bytes again, so re-running the reassembly is the only fix.
            raise HTTPException(
                status_code=404,
                detail="This image is no longer cached -- reassemble the file again.",
            )
        _cached_images.move_to_end(token)

        return Response(
            jpg_bytes,
            media_type="image/jpeg",
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
        PICAM_IMAGE_PATH + "/{token}/{filename}", picam_image, methods=["GET"]
    )
