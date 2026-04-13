"""
Pool renderer — composites a photorealistic pool into the backyard.

Real mode (pick one via RENDER_PROVIDER in .env):
  - "fal"       → fal.ai FLUX Fill Pro  (recommended, best quality)
  - "openai"    → OpenAI gpt-image-1 edits  (if you have GPT-4o access)
  - "replicate" → Replicate SD inpainting  (original option)
  - "stability" → Stability AI API directly

Demo mode: draws a clean pool using PIL (no API needed).
"""
import os
import asyncio
import base64
import io
from PIL import Image, ImageDraw, ImageFilter
import numpy as np

import config
import logging

logger = logging.getLogger(__name__)

POOL_PROMPT = (
    "photorealistic luxury swimming pool with white travertine pool deck and coping stones, "
    "turquoise blue water, aerial top-down satellite view, Florida backyard, "
    "professional real estate photography, hyper realistic, 4k"
)
POOL_NEGATIVE = "cartoon, sketch, blurry, distorted, ugly, low quality, people"


def _img_to_b64_uri(pil_img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    mime = "image/png" if fmt == "PNG" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode()


def _build_mask(img: Image.Image, pool_zone: dict, pad: int = 10) -> Image.Image:
    """White rectangle over pool zone, black everywhere else."""
    x1, y1, x2, y2 = pool_zone["x1"], pool_zone["y1"], pool_zone["x2"], pool_zone["y2"]
    mask = Image.new("RGB", img.size, "black")
    ImageDraw.Draw(mask).rectangle([x1 - pad, y1 - pad, x2 + pad, y2 + pad], fill="white")
    return mask




def create_reveal_gif(satellite_path: str, rendered_path: str, prospect_id: str, pool_zone: dict = None) -> str:
    """
    Multi-stage pool construction animation:
      1. Satellite hold (2s)
      2. Yellow outline draws around pool zone
      3. Brown excavation fills pool zone
      4. Blue water rises from bottom
      5. Crossfade to AI render
    Uses direct RGB drawing to avoid GIF palette quantization issues.
    """
    import os
    GIF_W, GIF_H = 800, 600

    out_path = satellite_path.replace("_satellite.jpg", "_reveal.gif")
    if not os.path.exists(satellite_path):
        logger.warning(f"Satellite image not found: {satellite_path}")
        return ""

    # Load and resize satellite image
    sat = Image.open(satellite_path).convert("RGB").resize((GIF_W, GIF_H), Image.LANCZOS)

    # Load rendered image if available
    rendered = None
    if rendered_path and os.path.exists(rendered_path):
        rendered = Image.open(rendered_path).convert("RGB").resize((GIF_W, GIF_H), Image.LANCZOS)

    # Scale pool zone from 640x480 to 800x600
    sx = GIF_W / 640
    sy = GIF_H / 480
    if pool_zone:
        px1 = int(pool_zone["x1"] * sx)
        py1 = int(pool_zone["y1"] * sy)
        px2 = int(pool_zone["x2"] * sx)
        py2 = int(pool_zone["y2"] * sy)
    else:
        px1 = int(GIF_W * 0.30)
        py1 = int(GIF_H * 0.55)
        px2 = int(GIF_W * 0.70)
        py2 = int(GIF_H * 0.82)

    frames = []
    durations = []

    def add_frame(img, ms):
        frames.append(img.copy())
        durations.append(ms)

    # Stage 1: Hold on satellite (8 frames x 250ms = 2s)
    for _ in range(8):
        add_frame(sat, 250)

    # Stage 2: Yellow outline draws progressively (14 frames)
    outline_color = (255, 220, 50)
    outline_width = 4
    perimeter_pts = []
    for x in range(px1, px2):
        perimeter_pts.append((x, py1))
    for y in range(py1, py2):
        perimeter_pts.append((px2, y))
    for x in range(px2, px1, -1):
        perimeter_pts.append((x, py2))
    for y in range(py2, py1, -1):
        perimeter_pts.append((px1, y))

    n_outline_frames = 14
    pts_per_frame = max(1, len(perimeter_pts) // n_outline_frames)
    for i in range(n_outline_frames):
        frame = sat.copy()
        batch_end = min(len(perimeter_pts), (i + 1) * pts_per_frame)
        drawn = perimeter_pts[:batch_end]
        for (ppx, ppy) in drawn:
            for dx in range(-outline_width // 2, outline_width // 2 + 1):
                for dy in range(-outline_width // 2, outline_width // 2 + 1):
                    nx, ny = ppx + dx, ppy + dy
                    if 0 <= nx < GIF_W and 0 <= ny < GIF_H:
                        frame.putpixel((nx, ny), outline_color)
        add_frame(frame, 80)

    # Stage 3: Brown excavation fills pool zone top-to-bottom (16 frames)
    dig_color = (139, 100, 60)
    n_dig = 16
    pool_h = py2 - py1
    for i in range(n_dig):
        frame = sat.copy()
        draw = ImageDraw.Draw(frame)
        draw.rectangle([px1, py1, px2, py2], outline=outline_color, width=outline_width)
        fill_bottom = py1 + int(pool_h * (i + 1) / n_dig)
        if fill_bottom > py1 + outline_width:
            draw.rectangle([px1 + outline_width, py1 + outline_width,
                            px2 - outline_width, fill_bottom], fill=dig_color)
        add_frame(frame, 80)

    # Stage 4: Blue water rises from bottom (16 frames)
    water_color = (0, 110, 190)
    water_shimmer = (30, 150, 230)
    n_water = 16
    for i in range(n_water):
        frame = sat.copy()
        draw = ImageDraw.Draw(frame)
        draw.rectangle([px1 + outline_width, py1 + outline_width,
                        px2 - outline_width, py2 - outline_width], fill=dig_color)
        draw.rectangle([px1, py1, px2, py2], outline=outline_color, width=outline_width)
        water_top = py2 - int(pool_h * (i + 1) / n_water)
        water_top = max(py1 + outline_width, water_top)
        draw.rectangle([px1 + outline_width, water_top,
                        px2 - outline_width, py2 - outline_width], fill=water_color)
        if water_top + 3 < py2:
            draw.line([(px1 + outline_width, water_top + 2),
                       (px2 - outline_width, water_top + 2)],
                      fill=water_shimmer, width=2)
        add_frame(frame, 80)

    # Stage 5: Hold on finished pool (4 frames x 300ms)
    final_state = sat.copy()
    fd = ImageDraw.Draw(final_state)
    fd.rectangle([px1 + outline_width, py1 + outline_width,
                  px2 - outline_width, py2 - outline_width], fill=dig_color)
    fd.rectangle([px1, py1, px2, py2], outline=outline_color, width=outline_width)
    fd.rectangle([px1 + outline_width, py1 + outline_width,
                  px2 - outline_width, py2 - outline_width], fill=water_color)
    fd.line([(px1 + outline_width, py1 + outline_width + 2),
             (px2 - outline_width, py1 + outline_width + 2)],
            fill=water_shimmer, width=2)
    for _ in range(4):
        add_frame(final_state, 300)

    # Stage 6: Crossfade to rendered image (12 frames) then hold (6 frames)
    if rendered:
        n_fade = 12
        for i in range(n_fade):
            alpha = (i + 1) / n_fade
            blended = Image.blend(final_state, rendered, alpha)
            add_frame(blended, 80)
        for _ in range(6):
            add_frame(rendered, 400)

    if not frames:
        return ""

    frames[0].save(
        out_path,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=False,
    )
    logger.info(f"Reveal GIF saved: {out_path} ({len(frames)} frames)")
    return out_path


async def render_pool(prospect_id: str, satellite_path: str, pool_zone: dict) -> str:
    out_path = os.path.join(config.IMAGES_DIR, f"{prospect_id}_rendered.jpg")
    provider = config.RENDER_PROVIDER.lower()

    use_real = not config.DEMO_MODE and (
        config.FAL_API_KEY or config.REPLICATE_API_TOKEN or
        config.OPENAI_API_KEY or config.STABILITY_API_KEY
    )

    if not use_real:
        await _demo_render(satellite_path, out_path, pool_zone)
    else:
        try:
            if provider == "fal" and config.FAL_API_KEY:
                await _fal_render(satellite_path, out_path, pool_zone)
            elif provider == "openai" and config.OPENAI_API_KEY:
                await _openai_render(satellite_path, out_path, pool_zone)
            elif provider == "stability" and config.STABILITY_API_KEY:
                await _stability_render(satellite_path, out_path, pool_zone)
            elif config.REPLICATE_API_TOKEN:
                await _replicate_render(satellite_path, out_path, pool_zone)
            else:
                await _demo_render(satellite_path, out_path, pool_zone)
        except Exception as e:
            logger.warning(f"AI render failed ({e}), falling back to demo render")
            await _demo_render(satellite_path, out_path, pool_zone)


    # Generate reveal GIF (satellite → pool animation)
    gif_path = create_reveal_gif(satellite_path, out_path, prospect_id, pool_zone)
    logger.info(f"GIF generated: {gif_path}")
    return out_path


# ── fal.ai — FLUX Fill Pro (recommended) ─────────────────────────────────────

async def _fal_render(satellite_path: str, out_path: str, pool_zone: dict):
    """
    fal.ai FLUX Fill Pro inpainting.
    Get key at: https://fal.ai/dashboard/keys
    Docs: https://fal.ai/models/fal-ai/flux-pro/v1/fill
    """
    import fal_client
    import httpx

    os.environ["FAL_KEY"] = config.FAL_API_KEY

    img = Image.open(satellite_path).convert("RGB")
    mask = _build_mask(img, pool_zone)

    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: fal_client.run(
            "fal-ai/flux-pro/v1/fill",
            arguments={
                "prompt": POOL_PROMPT,
                "image_url": _img_to_b64_uri(img),
                "mask_url": _img_to_b64_uri(mask),
                "num_inference_steps": 28,
                "guidance_scale": 3.5,
                "output_format": "jpeg",
            },
        ),
    )

    image_url = result["images"][0]["url"]
    async with httpx.AsyncClient() as client:
        resp = await client.get(image_url)
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(resp.content)


# ── OpenAI gpt-image-1 edits ─────────────────────────────────────────────────

async def _openai_render(satellite_path: str, out_path: str, pool_zone: dict):
    """
    OpenAI image edits API (gpt-image-1).
    Get key at: https://platform.openai.com/api-keys
    """
    from openai import AsyncOpenAI
    import httpx

    client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)

    img = Image.open(satellite_path).convert("RGBA").resize((1024, 1024))
    mask = _build_mask(img.convert("RGB"), pool_zone).convert("RGBA")
    # OpenAI mask: transparent = edit here
    mask_rgba = Image.new("RGBA", img.size, (0, 0, 0, 255))
    x1, y1 = pool_zone["x1"], pool_zone["y1"]
    x2, y2 = pool_zone["x2"], pool_zone["y2"]
    for px in range(max(0, x1-10), min(img.width, x2+10)):
        for py in range(max(0, y1-10), min(img.height, y2+10)):
            mask_rgba.putpixel((px, py), (0, 0, 0, 0))

    img_buf = io.BytesIO(); img.save(img_buf, "PNG"); img_buf.seek(0)
    mask_buf = io.BytesIO(); mask_rgba.save(mask_buf, "PNG"); mask_buf.seek(0)

    response = await client.images.edit(
        model="gpt-image-1",
        image=("image.png", img_buf, "image/png"),
        mask=("mask.png", mask_buf, "image/png"),
        prompt=POOL_PROMPT,
        n=1,
        size="1024x1024",
    )

    img_b64 = response.data[0].b64_json
    img_bytes = base64.b64decode(img_b64)
    result_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    result_img.save(out_path, "JPEG", quality=95)


# ── Stability AI direct API ───────────────────────────────────────────────────

async def _stability_render(satellite_path: str, out_path: str, pool_zone: dict):
    """
    Stability AI inpainting REST API.
    Get key at: https://platform.stability.ai/account/keys
    """
    import httpx

    img = Image.open(satellite_path).convert("RGB").resize((768, 512))
    mask = _build_mask(img, pool_zone)

    img_buf = io.BytesIO(); img.save(img_buf, "PNG"); img_buf.seek(0)
    mask_buf = io.BytesIO(); mask.save(mask_buf, "PNG"); mask_buf.seek(0)

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/image-to-image/masking",
            headers={"Authorization": f"Bearer {config.STABILITY_API_KEY}", "Accept": "image/png"},
            data={
                "text_prompts[0][text]": POOL_PROMPT,
                "text_prompts[0][weight]": "1",
                "text_prompts[1][text]": POOL_NEGATIVE,
                "text_prompts[1][weight]": "-1",
                "mask_source": "MASK_IMAGE_WHITE",
                "cfg_scale": "7",
                "steps": "30",
            },
            files={
                "init_image": ("image.png", img_buf, "image/png"),
                "mask_image": ("mask.png", mask_buf, "image/png"),
            },
        )
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(resp.content)


# ── Replicate (original) ──────────────────────────────────────────────────────

async def _replicate_render(satellite_path: str, out_path: str, pool_zone: dict):
    """Replicate Stable Diffusion inpainting."""
    import replicate
    import httpx

    img = Image.open(satellite_path).convert("RGB")
    mask = _build_mask(img, pool_zone)

    output = replicate.run(
        "stability-ai/stable-diffusion-inpainting:95b7223104132402a9ae91cc677285bc5eb997834bd2349fa486f53910fd68b3",
        input={
            "prompt": POOL_PROMPT,
            "negative_prompt": POOL_NEGATIVE,
            "image": _img_to_b64_uri(img),
            "mask": _img_to_b64_uri(mask),
            "num_inference_steps": 30,
            "guidance_scale": 7.5,
        },
    )

    async with httpx.AsyncClient() as client:
        resp = await client.get(str(output[0]))
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(resp.content)


async def _demo_render(satellite_path: str, out_path: str, pool_zone: dict):
    """
    Draw a convincing pool into the satellite image using PIL.
    No AI API required — good enough for demos and testing.
    """
    await asyncio.sleep(1.5)  # Simulate rendering time

    img = Image.open(satellite_path).convert("RGBA")
    W, H = img.size
    x1, y1, x2, y2 = pool_zone["x1"], pool_zone["y1"], pool_zone["x2"], pool_zone["y2"]

    pool_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(pool_layer)

    # ── Pool deck / coping (concrete surround) ──
    deck_pad = 14
    draw.rounded_rectangle(
        [x1 - deck_pad, y1 - deck_pad, x2 + deck_pad, y2 + deck_pad],
        radius=8,
        fill=(195, 185, 170, 255),  # concrete color
    )

    # ── Pool coping edge (lighter) ──
    coping_pad = 6
    draw.rounded_rectangle(
        [x1 - coping_pad, y1 - coping_pad, x2 + coping_pad, y2 + coping_pad],
        radius=6,
        fill=(215, 210, 200, 255),
    )

    # ── Pool water (blue gradient effect) ──
    pool_w = x2 - x1
    pool_h = y2 - y1

    # Draw water with slight color variation for depth
    for row in range(pool_h):
        t = row / pool_h
        # Shallow end lighter, deep end darker
        r = int(30 + 20 * t)
        g = int(140 + 30 * (1 - t))
        b = int(190 + 40 * (1 - t))
        draw.line([(x1, y1 + row), (x2, y1 + row)], fill=(r, g, b, 245))

    # Pool lane lines (subtle)
    lane_color = (80, 170, 220, 100)
    num_lanes = 3
    for i in range(1, num_lanes):
        lx = x1 + (pool_w * i // num_lanes)
        draw.line([(lx, y1 + 4), (lx, y2 - 4)], fill=lane_color, width=1)

    # Pool steps in corner
    step_size = 10
    for s in range(3):
        offset = s * step_size
        draw.rectangle(
            [x1 + offset, y1 + offset, x1 + step_size * 4 - offset, y1 + step_size * 2],
            fill=(160, 195, 215, 200),
        )

    # Slight water shimmer highlight
    shimmer_x = x1 + pool_w // 3
    shimmer_y = y1 + pool_h // 3
    draw.ellipse(
        [shimmer_x - 8, shimmer_y - 3, shimmer_x + 8, shimmer_y + 3],
        fill=(255, 255, 255, 60),
    )

    # Blur pool layer edges for realistic blending
    pool_layer_blurred = pool_layer.filter(ImageFilter.GaussianBlur(0.8))

    # Composite onto satellite image
    composite = Image.alpha_composite(img, pool_layer_blurred)
    composite = composite.convert("RGB")

    # Add outdoor furniture (lounge chairs on deck)
    draw2 = ImageDraw.Draw(composite)
    chair_y = y2 + deck_pad - 8
    for cx in [x1 + 10, x1 + 25]:
        draw2.rectangle([cx, chair_y, cx + 6, chair_y + 14], fill=(180, 140, 80))

    composite.save(out_path, "JPEG", quality=95)
