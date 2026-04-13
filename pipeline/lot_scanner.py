"""
Lot scanner — downloads aerial/satellite imagery and identifies the pool-ready zone.

Real mode:
  1. Tries Google Aerial View API for photorealistic cinematic aerial video (extracts frame)
  2. Falls back to Google Maps Static API (satellite, scale=2, 1280x960 hi-res) if aerial unavailable
  3. Uses OpenAI Vision (GPT-4o) to intelligently identify the backyard pool zone
Demo mode: generates a synthetic image using PIL.
"""
import os
import base64
import json
import re
import subprocess
import tempfile
import asyncio
import logging
import httpx
import numpy as np
from PIL import Image, ImageDraw
import io
from urllib.parse import quote

import config

logger = logging.getLogger(__name__)


async def get_satellite_image(prospect_id: str, lat: float, lng: float, address: str) -> str:
    """Download aerial/satellite imagery for the property. Returns local file path."""
    out_path = os.path.join(config.IMAGES_DIR, f"{prospect_id}_satellite.jpg")

    if config.DEMO_MODE or not config.GOOGLE_MAPS_API_KEY:
        await _generate_demo_satellite(out_path, address)
    else:
        try:
            # Try Google Aerial View API first (photorealistic 3D cinematic)
            success = await _fetch_aerial_view(out_path, address)
            if not success:
                # Fall back to high-res satellite (scale=2)
                await _fetch_google_satellite(out_path, lat, lng)
        except Exception as e:
            logger.warning(f"Aerial/satellite fetch failed ({e}), falling back to demo")
            await _generate_demo_satellite(out_path, address)

    return out_path


async def _fetch_aerial_view(out_path: str, address: str) -> bool:
    """
    Fetch photorealistic aerial video from Google Aerial View API.
    Extracts a clean mid-flyover frame and saves as JPEG.
    Returns True if successful, False if coverage not available.
    """
    api_key = config.GOOGLE_MAPS_API_KEY
    encoded_address = quote(address)
    lookup_url = (
        f"https://aerialview.googleapis.com/v1/videos:lookupVideo"
        f"?address={encoded_address}&key={api_key}"
    )

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(lookup_url)
        if resp.status_code != 200:
            logger.warning(f"Aerial View API HTTP {resp.status_code}")
            return False
        data = resp.json()

    state = data.get("state", "UNKNOWN")
    logger.info(f"Aerial View API state for '{address}': {state}")

    if state != "PROCESSED":
        logger.info(f"Aerial view not available (state={state}), using satellite fallback")
        return False

    uris = data.get("uris", {})
    video_url = uris.get("landscapeUri") or uris.get("portraitUri")
    if not video_url:
        logger.warning("Aerial View: no video URI in response")
        return False

    # Download video
    async with httpx.AsyncClient(timeout=45, follow_redirects=True) as client:
        video_resp = await client.get(video_url)
        video_bytes = video_resp.content

    if len(video_bytes) < 5000:
        logger.warning(f"Aerial video too small ({len(video_bytes)} bytes)")
        return False

    logger.info(f"Aerial video downloaded: {len(video_bytes) // 1024} KB")

    # Write to temp file, extract frame with ffmpeg
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_vid:
        tmp_vid.write(video_bytes)
        tmp_vid_path = tmp_vid.name

    try:
        def _run_ffmpeg(seek_sec):
            return subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-ss", str(seek_sec),
                    "-i", tmp_vid_path,
                    "-vframes", "1",
                    "-vf", "scale=640:480:force_original_aspect_ratio=decrease,"
                           "pad=640:480:(ow-iw)/2:(oh-ih)/2:black",
                    "-q:v", "2",
                    out_path,
                ],
                capture_output=True,
                timeout=30,
            )

        # Try mid-point of flyover first (best angle), then fall back to start
        result = _run_ffmpeg(2)
        if result.returncode != 0 or not os.path.exists(out_path) or os.path.getsize(out_path) < 1000:
            result = _run_ffmpeg(0)

        success = result.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 1000
        if success:
            logger.info(f"Aerial frame extracted successfully: {out_path}")
        else:
            logger.warning(f"ffmpeg extraction failed: {result.stderr.decode()[:200]}")
        return success

    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning(f"ffmpeg unavailable ({e}), aerial view skipped")
        return False
    finally:
        try:
            os.unlink(tmp_vid_path)
        except Exception:
            pass


async def _fetch_google_satellite(out_path: str, lat: float, lng: float):
    """Fetch high-res satellite image from Google Maps Static API (scale=2 = 1280x960)."""
    url = "https://maps.googleapis.com/maps/api/staticmap"
    params = {
        "center": f"{lat},{lng}",
        "zoom": "19",
        "size": "640x480",
        "scale": "2",
        "maptype": "satellite",
        "key": config.GOOGLE_MAPS_API_KEY,
    }
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        img_data = resp.content

    img = Image.open(io.BytesIO(img_data)).convert("RGB")
    # Resize to 640x480 for consistent pipeline dimensions
    img = img.resize((640, 480), Image.LANCZOS)
    img.save(out_path, "JPEG", quality=92)
    logger.info(f"Hi-res satellite saved (scale=2 downsampled to 640x480): {out_path}")


async def identify_pool_zone(satellite_path: str, address: str = "", lot_sqft: int = 15000) -> dict:
    """
    Identify the best pool zone in the backyard using OpenAI Vision.
    Falls back to heuristic if Vision unavailable or fails.
    """
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if openai_key and not config.DEMO_MODE:
        try:
            zone = await _vision_detect_pool_zone(satellite_path, address, openai_key)
            if zone:
                logger.info(f"Vision pool zone: {zone['pool_zone']} ({zone.get('vision_confidence','?')} confidence)")
                return zone
        except Exception as e:
            logger.warning(f"Vision pool zone detection failed ({e}), using heuristic")

    return _heuristic_pool_zone(satellite_path, lot_sqft)


async def _vision_detect_pool_zone(satellite_path: str, address: str, openai_key: str) -> dict | None:
    """Use GPT-4o vision to identify the backyard and optimal pool placement."""
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=openai_key)

    with open(satellite_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    img = Image.open(satellite_path)
    W, H = img.size

    prompt = f"""This is a satellite or aerial image of a residential property at: {address}
Image dimensions: {W}x{H} pixels.

Task: Identify the BACKYARD and find the best rectangular area where a swimming pool could realistically be installed.

Guidelines:
- Backyard is typically the area BEHIND the main house structure
- Avoid placing the pool over: roof, driveway, obvious trees/large shrubs, neighboring lots
- Target open grass or lawn area
- Realistic pool size at this scale: roughly 50-120 pixels wide, 30-80 pixels tall
- If this looks like an aerial/angled view, estimate accordingly

Return ONLY valid JSON, no other text:
{{"x1": <int>, "y1": <int>, "x2": <int>, "y2": <int>, "confidence": "high"|"medium"|"low", "note": "<what you see>"}}

x1,y1 = top-left of pool zone. x2,y2 = bottom-right. Pixel coordinates."""

    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_b64}",
                        "detail": "high",
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }],
        max_tokens=250,
        temperature=0,
    )

    raw = response.choices[0].message.content.strip()
    logger.info(f"GPT-4o vision response: {raw}")

    match = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
    if not match:
        return None

    data = json.loads(match.group())
    x1 = int(data["x1"])
    y1 = int(data["y1"])
    x2 = int(data["x2"])
    y2 = int(data["y2"])

    # Clamp to image bounds with minimum pool size
    margin = 10
    x1 = max(margin, min(x1, W - margin - 40))
    y1 = max(margin, min(y1, H - margin - 20))
    x2 = max(x1 + 40, min(x2, W - margin))
    y2 = max(y1 + 20, min(y2, H - margin))

    pool_w_px = x2 - x1
    pool_h_px = y2 - y1
    # At zoom 19 satellite: ~0.30 m/px -> 1 ft ~= 3.28 px
    px_per_ft = 3.28 / 0.30
    pool_sqft = int((pool_w_px / px_per_ft) * (pool_h_px / px_per_ft))
    pool_sqft = max(150, min(pool_sqft, 900))

    return {
        "pool_zone": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "pool_sqft": pool_sqft,
        "backyard_sqft": 12000,
        "setbacks_clear": True,
        "lot_sqft": 15000,
        "detection_method": "openai_vision",
        "vision_confidence": data.get("confidence", "medium"),
        "vision_note": data.get("note", ""),
    }


def _heuristic_pool_zone(satellite_path: str, lot_sqft: int) -> dict:
    """Fallback heuristic: place pool in lower-center of image."""
    try:
        img = Image.open(satellite_path)
        W, H = img.size
    except Exception:
        W, H = 640, 480

    scale = W / 640
    backyard_y_start = int(H * 0.50)
    cx = W // 2
    cy = int(H * 0.70)
    pool_w = int(90 * scale)
    pool_h = int(45 * scale)

    x1 = max(20, cx - pool_w // 2)
    y1 = max(backyard_y_start, cy - pool_h // 2)
    x2 = min(W - 20, x1 + pool_w)
    y2 = min(H - 20, y1 + pool_h)

    pool_sqft = int((x2 - x1) / scale * (y2 - y1) / scale)

    return {
        "pool_zone": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "pool_sqft": pool_sqft,
        "backyard_sqft": lot_sqft - 3000,
        "setbacks_clear": True,
        "lot_sqft": lot_sqft,
        "detection_method": "heuristic",
    }


async def _generate_demo_satellite(out_path: str, address: str):
    """Generate a synthetic aerial-looking image for demo mode."""
    W, H = 640, 480
    img = Image.new("RGB", (W, H), (85, 120, 65))

    rng = np.random.default_rng(abs(hash(address)) % (2**32))
    noise = rng.integers(0, 25, (H, W, 3), dtype=np.uint8)
    grass = np.array(img) + noise - 12
    grass = np.clip(grass, 0, 255).astype(np.uint8)
    img = Image.fromarray(grass)

    draw = ImageDraw.Draw(img)

    # House footprint (upper-center)
    house_x1, house_y1 = W // 2 - 80, H // 4 - 40
    house_x2, house_y2 = W // 2 + 80, H // 4 + 60
    draw.rectangle([house_x1, house_y1, house_x2, house_y2], fill=(200, 190, 175))
    draw.line([(W // 2, house_y1), (W // 2, house_y2)], fill=(160, 150, 135), width=3)

    # Driveway
    drive_x = W // 2 + 60
    draw.polygon([
        (drive_x, house_y2),
        (drive_x + 30, house_y2),
        (drive_x + 50, H - 20),
        (drive_x - 20, H - 20),
    ], fill=(140, 135, 120))

    # Trees
    for tx, ty, tr in [(120, 100, 28), (500, 80, 22), (80, 300, 24), (560, 320, 20)]:
        draw.ellipse([tx - tr, ty - tr, tx + tr, ty + tr], fill=(34, 85, 34))
        draw.ellipse([tx - tr + 5, ty - tr + 5, tx + tr - 5, ty + tr - 5], fill=(45, 100, 45))

    # Fence (backyard boundary)
    draw.line([(house_x1, house_y2 + 10), (house_x1, H - 30)], fill=(160, 140, 100), width=2)
    draw.line([(house_x2, house_y2 + 10), (house_x2, H - 30)], fill=(160, 140, 100), width=2)
    draw.line([(house_x1, H - 30), (house_x2, H - 30)], fill=(160, 140, 100), width=2)

    img.save(out_path, "JPEG", quality=92)
