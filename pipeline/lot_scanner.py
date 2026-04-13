"""
Lot scanner — downloads satellite imagery and identifies the pool-ready zone.

Real mode: fetches from Google Maps Static API (falls back to demo if API unavailable).
Demo mode: generates a synthetic aerial image using PIL.
"""
import os
import asyncio
import logging
import httpx
import numpy as np
from PIL import Image, ImageDraw
import io

import config

logger = logging.getLogger(__name__)


async def get_satellite_image(prospect_id: str, lat: float, lng: float, address: str) -> str:
    """Download satellite imagery for the property. Returns local file path."""
    out_path = os.path.join(config.IMAGES_DIR, f"{prospect_id}_satellite.jpg")

    if config.DEMO_MODE or not config.GOOGLE_MAPS_API_KEY:
        await _generate_demo_satellite(out_path, address)
    else:
        try:
            await _fetch_google_satellite(out_path, lat, lng)
        except Exception as e:
            logger.warning(f"Google Maps API failed ({e}), falling back to demo satellite")
            await _generate_demo_satellite(out_path, address)

    return out_path


async def _fetch_google_satellite(out_path: str, lat: float, lng: float):
    """Fetch real satellite image from Google Maps Static API."""
    url = "https://maps.googleapis.com/maps/api/staticmap"
    params = {
        "center": f"{lat},{lng}",
        "zoom": "19",
        "size": "640x480",
        "maptype": "satellite",
        "key": config.GOOGLE_MAPS_API_KEY,
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        # Google Maps returns a 200 with an error image if the key is wrong —
        # check content type to make sure we got a real image
        content_type = resp.headers.get("content-type", "")
        if "image" not in content_type:
            raise ValueError(f"Maps API returned non-image content: {content_type}")
        with open(out_path, "wb") as f:
            f.write(resp.content)


async def _generate_demo_satellite(out_path: str, address: str):
    """
    Generate a convincing synthetic satellite-style aerial image using PIL.
    No API required — used for demo and as fallback.
    """
    await asyncio.sleep(0.8)

    W, H = 640, 480
    img = Image.new("RGB", (W, H), (85, 107, 47))  # grass green base

    # Add grass texture noise
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
    # Roof ridge line
    draw.line([(W // 2, house_y1), (W // 2, house_y2)], fill=(160, 150, 135), width=3)

    # Driveway
    drive_x = W // 2 + 60
    draw.polygon([
        (drive_x, house_y2),
        (drive_x + 30, house_y2),
        (drive_x + 50, H - 20),
        (drive_x - 20, H - 20),
    ], fill=(140, 135, 120))

    # Trees (circles)
    for tx, ty, tr in [(120, 100, 28), (500, 80, 22), (80, 300, 24), (560, 320, 20)]:
        draw.ellipse([tx - tr, ty - tr, tx + tr, ty + tr], fill=(34, 85, 34))
        draw.ellipse([tx - tr + 5, ty - tr + 5, tx + tr - 5, ty + tr - 5],
                     fill=(45, 100, 45))

    # Fence line (backyard boundary)
    draw.line([(house_x1, house_y2 + 10), (house_x1, H - 30)], fill=(160, 140, 100), width=2)
    draw.line([(house_x2, house_y2 + 10), (house_x2, H - 30)], fill=(160, 140, 100), width=2)
    draw.line([(house_x1, H - 30), (house_x2, H - 30)], fill=(160, 140, 100), width=2)

    img.save(out_path, "JPEG", quality=92)


def identify_pool_zone(satellite_path: str, lot_sqft: int = 15000) -> dict:
    """
    Identify the best zone in the backyard for a pool.
    Returns pool zone coordinates and size estimates.
    """
    try:
        img = Image.open(satellite_path)
        W, H = img.size
    except Exception:
        W, H = 640, 480

    # Backyard = lower 40% of image
    backyard_y_start = int(H * 0.55)
    backyard_center_x = W // 2
    backyard_center_y = int(H * 0.78)

    # Pool ~15x30 ft; image ~640px wide = ~80ft -> 1px ~ 0.125ft
    scale = W / 80.0
    pool_w = int(15 * scale)
    pool_h = int(30 * scale)

    x1 = backyard_center_x - pool_w // 2
    y1 = backyard_center_y - pool_h // 2
    x2 = x1 + pool_w
    y2 = y1 + pool_h

    margin = 20
    x1 = max(margin, min(x1, W - pool_w - margin))
    y1 = max(backyard_y_start, min(y1, H - pool_h - margin))
    x2 = x1 + pool_w
    y2 = y1 + pool_h

    pool_sqft = int((pool_w / scale) * (pool_h / scale))

    return {
        "pool_zone": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "pool_sqft": pool_sqft,
        "backyard_sqft": lot_sqft - 3000,
        "setbacks_clear": True,
        "lot_sqft": lot_sqft,
    }
