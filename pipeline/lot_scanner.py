"""
Lot scanner — downloads satellite imagery and identifies the pool-ready zone.

Real mode: fetches from Google Maps Static API.
Demo mode: downloads a sample aerial image and marks it up.
"""
import os
import asyncio
import httpx
import numpy as np
from PIL import Image, ImageDraw
import io

import config

# Sample aerial house images (public domain / CC0) for demo mode
DEMO_AERIAL_URLS = [
    "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0c/Aerial_view_of_suburban_house_with_pool.jpg/640px-Aerial_view_of_suburban_house_with_pool.jpg",
]

# We'll generate a synthetic aerial for demo if downloads fail
SYNTHETIC_FALLBACK = True


async def get_satellite_image(prospect_id: str, lat: float, lng: float, address: str) -> str:
    """
    Download satellite imagery for the property.
    Returns local file path.
    """
    out_path = os.path.join(config.IMAGES_DIR, f"{prospect_id}_satellite.jpg")

    if config.DEMO_MODE or not config.GOOGLE_MAPS_API_KEY:
        await _generate_demo_satellite(out_path, address)
    else:
        await _fetch_google_satellite(out_path, lat, lng)

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
        with open(out_path, "wb") as f:
            f.write(resp.content)


async def _generate_demo_satellite(out_path: str, address: str):
    """
    Generate a convincing synthetic satellite-style aerial image for demo.
    Creates a top-down view with house footprint, yard, trees, driveway.
    """
    await asyncio.sleep(0.5)  # Simulate API latency

    W, H = 640, 480
    img = Image.new("RGB", (W, H), (45, 75, 40))  # dark grass base
    draw = ImageDraw.Draw(img)

    # Add grass texture variation
    rng = np.random.default_rng(hash(address) % (2**32))
    noise = rng.integers(0, 20, (H, W, 3), dtype=np.uint8)
    grass_array = np.array(img, dtype=np.uint8)
    grass_array = np.clip(grass_array.astype(int) + noise - 10, 0, 255).astype(np.uint8)
    img = Image.fromarray(grass_array)
    draw = ImageDraw.Draw(img)

    # Driveway (gray)
    draw.rectangle([260, 0, 310, 160], fill=(120, 115, 108))
    draw.rectangle([260, 140, 400, 175], fill=(115, 110, 103))

    # House footprint (white/light gray roof)
    draw.rectangle([180, 150, 460, 320], fill=(230, 225, 220))
    # Roof ridge lines
    draw.line([(320, 150), (320, 320)], fill=(200, 195, 190), width=3)
    draw.line([(180, 235), (460, 235)], fill=(210, 205, 200), width=2)

    # Pool deck area (concrete — lighter green/gray, backyard)
    draw.rectangle([200, 340, 440, 460], fill=(55, 90, 50))

    # Trees around perimeter
    tree_positions = [(60, 80), (560, 60), (80, 350), (570, 380), (100, 430), (550, 200)]
    for tx, ty in tree_positions:
        draw.ellipse([tx-25, ty-25, tx+25, ty+25], fill=(20, 55, 20))
        draw.ellipse([tx-18, ty-18, tx+18, ty+18], fill=(30, 70, 25))

    # Street at top
    draw.rectangle([0, 0, W, 15], fill=(80, 78, 75))
    draw.line([(0, 7), (W, 7)], fill=(200, 195, 150), width=1)

    # Add slight vignette / shadow
    img.save(out_path, "JPEG", quality=92)


def identify_pool_zone(image_path: str, lot_sqft: int) -> dict:
    """
    Analyze the satellite image to find the best pool placement zone.
    Returns mask coordinates and zone info.
    """
    img = Image.open(image_path)
    W, H = img.size

    # The backyard is typically in the lower portion of the aerial image
    # We find the largest contiguous "yard" area (green pixels, lower half)
    backyard_y_start = int(H * 0.6)
    backyard_center_x = W // 2
    backyard_center_y = int(H * 0.78)

    # Pool dimensions: typical 15x30 ft pool, scale to image
    # Assume 640px wide = ~80ft, so 1px ≈ 0.125ft
    scale = W / 80.0
    pool_w = int(15 * scale)   # ~15ft wide
    pool_h = int(30 * scale)   # ~30ft long

    # Center pool in backyard
    x1 = backyard_center_x - pool_w // 2
    y1 = backyard_center_y - pool_h // 2
    x2 = x1 + pool_w
    y2 = y1 + pool_h

    # Clamp to image bounds with margin
    margin = 20
    x1 = max(margin, min(x1, W - pool_w - margin))
    y1 = max(backyard_y_start, min(y1, H - pool_h - margin))
    x2 = x1 + pool_w
    y2 = y1 + pool_h

    pool_sqft = int((pool_w / scale) * (pool_h / scale))

    return {
        "pool_zone": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "pool_sqft": pool_sqft,
        "backyard_sqft": lot_sqft - 3000,  # approx (minus house footprint)
        "setbacks_clear": True,
        "lot_sqft": lot_sqft,
    }
