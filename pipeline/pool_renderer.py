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




def _find_perspective_coeffs(src_corners, dst_corners):
    """Solve for 8-coefficient perspective transform (src->dst mapping)."""
    matrix = []
    for (x1, y1), (x2, y2) in zip(src_corners, dst_corners):
        matrix.append([x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1])
        matrix.append([0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1])
    A = np.array(matrix, dtype=np.float64)
    b = np.array([c for pair in dst_corners for c in pair], dtype=np.float64)
    coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return coeffs.tolist()


def _make_dirt_patch(w, h, rng):
    """Return an RGB numpy array with realistic excavation dirt texture."""
    base = np.array([101, 67, 33], dtype=np.float32)
    noise = rng.normal(0, 18, (h, w, 3)).astype(np.float32)
    clump = rng.integers(0, 2, (h // 6 + 1, w // 6 + 1)).astype(np.float32)
    clump = np.repeat(np.repeat(clump, 6, axis=0), 6, axis=1)[:h, :w, np.newaxis] * 20
    patch = np.clip(base + noise + clump, 20, 180).astype(np.uint8)
    return patch


def _make_concrete_patch(w, h, rng):
    """Return an RGB numpy array with rough concrete texture."""
    base = np.array([170, 168, 162], dtype=np.float32)
    noise = rng.normal(0, 12, (h, w, 3)).astype(np.float32)
    patch = np.clip(base + noise, 100, 220).astype(np.uint8)
    return patch


async def _replicate_construction_video(image_path: str) -> list:
    """
    Call Replicate Wan 2.1 image-to-video with a construction prompt.
    Returns a list of PIL RGB frames extracted from the output video.
    Falls back to empty list on any failure.
    """
    import glob
    import replicate
    import httpx
    import subprocess
    import tempfile

    prompt = (
        "Continue animating this exact aerial image. Do not change the house, the roof, the trees, "
        "the lawn, the fences, or any part of the existing property shown in the image. "
        "Keep every element of the scene identical to the input image — only add and animate a swimming "
        "pool being constructed in the backyard area of this specific property. "
        "Time-lapse construction sequence building on top of this image: a rectangular pit is excavated "
        "in the backyard, revealing dark brown clay soil with sharply defined edges. A compact excavator "
        "removes earth and piles displaced soil to one side. Construction workers spray white-gray gunite "
        "concrete along the pit walls and floor forming a clean pool shell, with a steel rebar grid "
        "briefly visible before concrete sets. Light gray concrete deck slabs form around all four sides. "
        "White coping tiles are placed along the perimeter edge. Crystal-clear water slowly rises from "
        "the bottom — first shallow aqua at the base, then bright turquoise halfway, then full cerulean "
        "blue sparkling water catching sunlight. Two white lounge chairs appear on the finished pool deck. "
        "The final frame shows this exact same property with a completed luxury rectangular swimming pool "
        "seamlessly integrated into the backyard, as if it was always there."
    )
    negative_prompt = (
        "different house, different property, different location, new scene, new background, "
        "change the house, move the trees, alter the landscape, replace the roof, "
        "cartoon, illustration, blur, watermark, text overlay, CGI look, unrealistic, "
        "low quality, glitch, distortion, overexposed, night, rain, snow, dead grass, deformed pool"
    )
    logger.info("Calling Replicate Wan 2.1 for construction video...")
    with open(image_path, "rb") as img_file:
        output = await asyncio.to_thread(
            replicate.run,
            "wavespeedai/wan-2.1-i2v-480p",
            input={
                "image": img_file,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_frames": 81,
                "guidance_scale": 7.5,
                "num_inference_steps": 30,
            },
        )

    # output may be a URL string or FileOutput object
    video_url = str(output) if not hasattr(output, "url") else output.url
    logger.info(f"Replicate video URL: {video_url}")

    # Download video
    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        resp = await client.get(video_url)
        video_bytes = resp.content

    if len(video_bytes) < 5000:
        logger.warning("Replicate video too small, skipping")
        return []

    # Write to temp file and extract frames at 10fps scaled to 800x600
    frames = []
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        import av
        container = av.open(tmp_path)
        video_stream = container.streams.video[0]
        fps = float(video_stream.average_rate) if video_stream.average_rate else 24.0
        sample_every = max(1, int(fps / 10))
        frame_count = 0
        for packet in container.demux(video_stream):
            for frame in packet.decode():
                if frame_count % sample_every == 0:
                    img = frame.to_image().convert("RGB")
                    img = img.resize((800, 600), Image.LANCZOS)
                    frames.append(img)
                frame_count += 1
        container.close()
        logger.info(f"Extracted {len(frames)} frames from Replicate video via PyAV")
    except Exception as e:
        logger.warning(f"Frame extraction error: {e}")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    return frames


async def create_reveal_gif(satellite_path: str, rendered_path: str, prospect_id: str, pool_zone: dict = None) -> str:
    """
    Cinematic pool construction GIF:
      1. Aerial overview hold (PIL)
      2. Drone zoom into pool zone - Ken Burns (PIL)
      3. Camera tilt to 45-degree perspective - QUAD warp (PIL)
      4. AI construction video - Replicate Wan 2.1 image-to-video
         (falls back to PIL dirt/concrete/water stages if Replicate unavailable)
      5. Hard cut to actual AI-rendered pool on the real property
    """
    GIF_W, GIF_H = 800, 600

    out_path = satellite_path.replace("_satellite.jpg", "_reveal.gif")
    if not os.path.exists(satellite_path):
        logger.warning(f"Satellite image not found: {satellite_path}")
        return ""

    rng = np.random.default_rng(42)

    sat = Image.open(satellite_path).convert("RGB").resize((GIF_W, GIF_H), Image.LANCZOS)

    rendered = None
    if rendered_path and os.path.exists(rendered_path):
        rendered = Image.open(rendered_path).convert("RGB").resize((GIF_W, GIF_H), Image.LANCZOS)

    sx, sy = GIF_W / 640, GIF_H / 480
    if pool_zone:
        px1 = int(pool_zone["x1"] * sx)
        py1 = int(pool_zone["y1"] * sy)
        px2 = int(pool_zone["x2"] * sx)
        py2 = int(pool_zone["y2"] * sy)
    else:
        px1, py1 = int(GIF_W * 0.30), int(GIF_H * 0.52)
        px2, py2 = int(GIF_W * 0.70), int(GIF_H * 0.80)

    pool_w = px2 - px1
    pool_h = py2 - py1
    cx, cy = (px1 + px2) // 2, (py1 + py2) // 2

    frames, durations = [], []

    def add(img, ms):
        frames.append(img.copy() if isinstance(img, Image.Image) else Image.fromarray(img))
        durations.append(ms)

    # ── Stage 1: Aerial overview hold ────────────────────────────────────────
    for _ in range(6):
        add(sat, 200)

    # ── Stage 2: Drone zoom (Ken Burns) ──────────────────────────────────────
    pad = max(pool_w, pool_h) * 1.4
    crop_end = (
        max(0, cx - pad), max(0, cy - pad),
        min(GIF_W, cx + pad), min(GIF_H, cy + pad),
    )
    n_zoom = 14
    for i in range(n_zoom):
        t = (i / (n_zoom - 1)) ** 2
        x0 = t * crop_end[0]
        y0 = t * crop_end[1]
        x1c = GIF_W + t * (crop_end[2] - GIF_W)
        y1c = GIF_H + t * (crop_end[3] - GIF_H)
        cropped = sat.crop((x0, y0, x1c, y1c)).resize((GIF_W, GIF_H), Image.LANCZOS)
        add(cropped, 60)

    zoomed = sat.crop(crop_end).resize((GIF_W, GIF_H), Image.LANCZOS)

    # ── Stage 3: Camera tilt to 45° (QUAD warp) ──────────────────────────────
    n_tilt = 12
    max_inset = int(GIF_W * 0.22)
    for i in range(n_tilt):
        t = (i / (n_tilt - 1)) ** 0.6
        inset = int(t * max_inset)
        quad_data = [inset, 0, GIF_W - inset, 0, GIF_W, GIF_H, 0, GIF_H]
        tilted = zoomed.transform((GIF_W, GIF_H), Image.QUAD, quad_data, Image.BILINEAR)
        add(tilted, 70)

    # Final tilted base frame — used as Replicate input
    quad_final = [max_inset, 0, GIF_W - max_inset, 0, GIF_W, GIF_H, 0, GIF_H]
    tilted_base = zoomed.transform((GIF_W, GIF_H), Image.QUAD, quad_final, Image.BILINEAR)

    # ── Stage 4: AI construction video via Replicate ─────────────────────────
    construction_frames = []
    replicate_token = os.environ.get("REPLICATE_API_TOKEN", "")
    if replicate_token and not config.DEMO_MODE:
        try:
            # Save tilted frame as input for Replicate
            tilted_input_path = satellite_path.replace("_satellite.jpg", "_tilted_input.jpg")
            tilted_base.save(tilted_input_path, "JPEG", quality=92)
            construction_frames = await _replicate_construction_video(tilted_input_path)
            try:
                os.unlink(tilted_input_path)
            except Exception:
                pass
        except Exception as e:
            logger.warning(f"Replicate construction video failed ({e}), using PIL fallback")
            construction_frames = []

    if construction_frames:
        # Use Replicate frames directly
        for frame in construction_frames:
            add(frame, 100)
    else:
        # ── PIL fallback: dirt → concrete → rebar → water ─────────────────
        tilted_arr = np.array(tilted_base)

        tilt_ratio_top = 1 - (1 - py1 / GIF_H) * (max_inset * 2 / GIF_W)
        tilt_ratio_bot = 1 - (1 - py2 / GIF_H) * (max_inset * 2 / GIF_W)
        tp_x1 = int(GIF_W / 2 - (pool_w / 2) * tilt_ratio_top)
        tp_x2 = int(GIF_W / 2 + (pool_w / 2) * tilt_ratio_top)
        tp_x1b = int(GIF_W / 2 - (pool_w / 2) * tilt_ratio_bot)
        tp_x2b = int(GIF_W / 2 + (pool_w / 2) * tilt_ratio_bot)
        tp_y1, tp_y2 = py1, py2

        dirt = _make_dirt_patch(pool_w, pool_h, rng)
        concrete = _make_concrete_patch(pool_w, pool_h, rng)

        # Excavation
        for i in range(12):
            t = (i + 1) / 12
            frame_arr = tilted_arr.copy()
            fill_rows = int(pool_h * t)
            for row in range(fill_rows):
                rt = row / pool_h
                xl = int(tp_x1 + (tp_x1b - tp_x1) * rt)
                xr = int(tp_x2 + (tp_x2b - tp_x2) * rt)
                yr = tp_y1 + row
                if 0 <= yr < GIF_H and xl < xr:
                    frame_arr[yr, xl:xr] = np.resize(dirt[row, :pool_w], (xr - xl, 3))
            add(frame_arr, 90)

        # Concrete + rebar
        wall = max(2, int(pool_w * 0.10))
        rebar_color = np.array([40, 35, 30], dtype=np.uint8)
        for i in range(10):
            t = (i + 1) / 10
            frame_arr = tilted_arr.copy()
            for row in range(pool_h):
                rt = row / pool_h
                xl = int(tp_x1 + (tp_x1b - tp_x1) * rt)
                xr = int(tp_x2 + (tp_x2b - tp_x2) * rt)
                yr = tp_y1 + row
                if 0 <= yr < GIF_H and xl < xr:
                    frame_arr[yr, xl:xr] = np.resize(dirt[row, :pool_w], (xr - xl, 3))
                    w_t = max(0, int(wall * t))
                    if w_t:
                        frame_arr[yr, xl:min(GIF_W, xl + w_t)] = np.resize(concrete[row, :w_t], (min(w_t, xr - xl), 3))
                        frame_arr[yr, max(0, xr - w_t):xr] = np.resize(concrete[row, :w_t], (min(w_t, xr - xl), 3))
            if t > 0.5:
                spacing = max(8, pool_h // 8)
                for row in range(pool_h):
                    rt = row / pool_h
                    xl = int(tp_x1 + (tp_x1b - tp_x1) * rt) + wall
                    xr = int(tp_x2 + (tp_x2b - tp_x2) * rt) - wall
                    yr = tp_y1 + row
                    if row % spacing == 0 and 0 <= yr < GIF_H and xl < xr:
                        frame_arr[yr, xl:xr] = rebar_color
            add(frame_arr, 90)

        # Water fill
        shimmer_off = 0
        for i in range(14):
            t = (i + 1) / 14
            frame_arr = tilted_arr.copy()
            water_start = int(pool_h * (1 - t))
            for row in range(pool_h):
                rt = row / pool_h
                xl = int(tp_x1 + (tp_x1b - tp_x1) * rt) + wall
                xr = int(tp_x2 + (tp_x2b - tp_x2) * rt) - wall
                yr = tp_y1 + row
                if 0 <= yr < GIF_H and xl < xr:
                    frame_arr[yr, xl:xr] = np.resize(concrete[row, :pool_w], (xr - xl, 3))
                    if row >= water_start:
                        depth = (row - water_start) / max(1, pool_h - water_start)
                        wc = np.array([int(depth * 10), int(80 + depth * 50), int(180 + depth * 40)], dtype=np.uint8)
                        frame_arr[yr, xl:xr] = wc
            if tp_y1 + water_start < GIF_H:
                yr_s = tp_y1 + water_start
                rt = water_start / pool_h
                xl = int(tp_x1 + (tp_x1b - tp_x1) * rt) + wall
                xr = int(tp_x2 + (tp_x2b - tp_x2) * rt) - wall
                shimmer = np.array([200, 230, 255], dtype=np.uint8)
                for sx in range(xl + shimmer_off % 12, xr, 20):
                    for sw in range(min(8, xr - sx)):
                        if 0 <= sx + sw < GIF_W:
                            frame_arr[yr_s, sx + sw] = shimmer
                shimmer_off += 4
            add(frame_arr, 80)

    # ── Stage 5: Hold on last construction frame ────────────────────────────
    # Hold on whatever the last frame is (final Replicate video frame or PIL water stage)
    # This is the most realistic "completed pool" view we have
    if frames:
        last_frame = frames[-1]
        for _ in range(8):
            add(last_frame, 400)

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
    logger.info(f"Cinematic GIF saved: {out_path} ({len(frames)} frames, ~{sum(durations)//1000}s)")
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
    gif_path = await create_reveal_gif(satellite_path, out_path, prospect_id, pool_zone)
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
