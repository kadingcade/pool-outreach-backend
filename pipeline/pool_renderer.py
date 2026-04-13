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


def create_reveal_gif(satellite_path: str, rendered_path: str, prospect_id: str, pool_zone: dict = None) -> str:
    """
    Cinematic pool construction time-lapse GIF:
      1. Aerial overview hold
      2. Drone zoom into pool zone (Ken Burns)
      3. Camera tilt to 45-degree perspective via QUAD warp
      4. Realistic excavation (dirt texture + soil pile)
      5. Concrete shell + rebar grid
      6. Water rising with shimmer
      7. Hard cut to actual AI-rendered pool on real property
    """
    import os
    GIF_W, GIF_H = 800, 600

    out_path = satellite_path.replace("_satellite.jpg", "_reveal.gif")
    if not os.path.exists(satellite_path):
        logger.warning(f"Satellite image not found: {satellite_path}")
        return ""

    rng = np.random.default_rng(42)

    # Load base images
    sat = Image.open(satellite_path).convert("RGB").resize((GIF_W, GIF_H), Image.LANCZOS)
    sat_arr = np.array(sat)

    rendered = None
    if rendered_path and os.path.exists(rendered_path):
        rendered = Image.open(rendered_path).convert("RGB").resize((GIF_W, GIF_H), Image.LANCZOS)

    # Scale pool zone from 640x480 to GIF size
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

    # ── Stage 1: Aerial overview hold (6 frames × 200ms) ──────────────────────
    for _ in range(6):
        add(sat, 200)

    # ── Stage 2: Drone zoom into pool zone (14 frames) ────────────────────────
    # Ken Burns: crop from full image down to ~2× pool zone padding
    pad = max(pool_w, pool_h) * 1.4
    crop_start = (0, 0, GIF_W, GIF_H)  # full image
    crop_end = (
        max(0, cx - pad),
        max(0, cy - pad),
        min(GIF_W, cx + pad),
        min(GIF_H, cy + pad),
    )
    n_zoom = 14
    for i in range(n_zoom):
        t = i / (n_zoom - 1)
        t = t * t  # ease-in
        x0 = crop_start[0] + t * (crop_end[0] - crop_start[0])
        y0 = crop_start[1] + t * (crop_end[1] - crop_start[1])
        x1c = crop_start[2] + t * (crop_end[2] - crop_start[2])
        y1c = crop_start[3] + t * (crop_end[3] - crop_start[3])
        cropped = sat.crop((x0, y0, x1c, y1c)).resize((GIF_W, GIF_H), Image.LANCZOS)
        add(cropped, 60)

    # Zoomed base used for tilt stages
    zoomed = sat.crop(crop_end).resize((GIF_W, GIF_H), Image.LANCZOS)

    # ── Stage 3: Camera tilt to 45° via QUAD warp (12 frames) ─────────────────
    # QUAD maps a source quadrilateral → fills destination rectangle.
    # Progressively inset the top edge to simulate camera tilting forward.
    n_tilt = 12
    max_inset = int(GIF_W * 0.22)
    for i in range(n_tilt):
        t = i / (n_tilt - 1)
        t = t ** 0.6  # ease-out
        inset = int(t * max_inset)
        # Source quad corners (top inset = perspective tilt)
        quad_data = [
            inset,         0,
            GIF_W - inset, 0,
            GIF_W,         GIF_H,
            0,             GIF_H,
        ]
        tilted = zoomed.transform((GIF_W, GIF_H), Image.QUAD, quad_data, Image.BILINEAR)
        add(tilted, 70)

    # Final tilted base frame (all construction stages rendered on this)
    final_tilt_inset = max_inset
    quad_final = [final_tilt_inset, 0, GIF_W - final_tilt_inset, 0, GIF_W, GIF_H, 0, GIF_H]
    tilted_base = zoomed.transform((GIF_W, GIF_H), Image.QUAD, quad_final, Image.BILINEAR)
    tilted_arr = np.array(tilted_base)

    # Recalculate pool zone coords in the tilted view
    # The tilt compresses top and keeps bottom: approximate new coords
    def tilt_y(y_orig):
        # map original y in zoomed image → tilted y
        # At top (y=0): pixels move inward. At bottom (y=H): unchanged.
        # Approximate inverse: compressed coord
        ratio = y_orig / GIF_H
        new_x_scale = 1 - (1 - ratio) * (final_tilt_inset * 2 / GIF_W)
        return y_orig, new_x_scale

    # Pool zone in tilted view (approximate)
    tilt_ratio_top = 1 - (1 - py1 / GIF_H) * (final_tilt_inset * 2 / GIF_W)
    tilt_ratio_bot = 1 - (1 - py2 / GIF_H) * (final_tilt_inset * 2 / GIF_W)
    tp_x1 = int(GIF_W / 2 - (pool_w / 2) * tilt_ratio_top)
    tp_x2 = int(GIF_W / 2 + (pool_w / 2) * tilt_ratio_top)
    tp_y1 = py1
    tp_x1b = int(GIF_W / 2 - (pool_w / 2) * tilt_ratio_bot)
    tp_x2b = int(GIF_W / 2 + (pool_w / 2) * tilt_ratio_bot)
    tp_y2 = py2

    # ── Stage 4: Excavation — dirt texture fills pool zone (14 frames) ─────────
    dirt = _make_dirt_patch(pool_w, pool_h, rng)
    n_dig = 14
    for i in range(n_dig):
        t = (i + 1) / n_dig
        frame_arr = tilted_arr.copy()
        # Fill pool zone progressively top-to-bottom with dirt
        fill_rows = int(pool_h * t)
        if fill_rows > 0:
            # Interpolate x bounds for trapezoidal pool zone in tilted view
            for row in range(fill_rows):
                row_t = row / pool_h
                x_left = int(tp_x1 + (tp_x1b - tp_x1) * row_t)
                x_right = int(tp_x2 + (tp_x2b - tp_x2) * row_t)
                y_row = tp_y1 + row
                if 0 <= y_row < GIF_H and x_left < x_right:
                    w_row = x_right - x_left
                    d_slice = dirt[row, :min(w_row, pool_w)]
                    if len(d_slice) < w_row:
                        d_slice = np.resize(d_slice, (w_row, 3))
                    frame_arr[y_row, x_left:x_right] = d_slice[:w_row]

        # Soil pile — mounded dirt to upper-right of pool
        pile_cx = min(GIF_W - 10, tp_x2 + 20)
        pile_cy = max(10, tp_y1 - 15)
        pile_r = int(pool_w * 0.18)
        for dy in range(-pile_r, pile_r + 1):
            for dx in range(-pile_r, pile_r + 1):
                dist = (dx ** 2 + dy ** 2) ** 0.5
                if dist < pile_r * t:
                    py_p = pile_cy + dy
                    px_p = pile_cx + dx
                    if 0 <= py_p < GIF_H and 0 <= px_p < GIF_W:
                        factor = 1 - dist / pile_r
                        earth = np.array([120 + int(20 * factor), 82 + int(15 * factor), 45], dtype=np.uint8)
                        frame_arr[py_p, px_p] = earth

        add(frame_arr, 90)

    # ── Stage 5: Concrete shell + rebar grid (10 frames) ──────────────────────
    concrete = _make_concrete_patch(pool_w, pool_h, rng)
    n_concrete = 10
    for i in range(n_concrete):
        t = (i + 1) / n_concrete
        frame_arr = tilted_arr.copy()
        # Full dirt first
        for row in range(pool_h):
            row_t = row / pool_h
            x_left = int(tp_x1 + (tp_x1b - tp_x1) * row_t)
            x_right = int(tp_x2 + (tp_x2b - tp_x2) * row_t)
            y_row = tp_y1 + row
            if 0 <= y_row < GIF_H and x_left < x_right:
                w_row = x_right - x_left
                d_slice = np.resize(dirt[row, :pool_w], (w_row, 3))
                frame_arr[y_row, x_left:x_right] = d_slice
        # Concrete walls build inward from edges
        wall = max(2, int(pool_w * 0.10 * t))
        for row in range(pool_h):
            row_t = row / pool_h
            x_left = int(tp_x1 + (tp_x1b - tp_x1) * row_t)
            x_right = int(tp_x2 + (tp_x2b - tp_x2) * row_t)
            y_row = tp_y1 + row
            if 0 <= y_row < GIF_H:
                # Left wall
                xl_end = min(GIF_W, x_left + wall)
                if xl_end > x_left:
                    w_s = xl_end - x_left
                    frame_arr[y_row, x_left:xl_end] = concrete[row, :w_s]
                # Right wall
                xr_start = max(0, x_right - wall)
                if xr_start < x_right:
                    w_s = x_right - xr_start
                    frame_arr[y_row, xr_start:x_right] = concrete[row, pool_w - w_s:pool_w]
        # Top / bottom walls
        for col_row in range(min(wall, pool_h)):
            row_t_top = col_row / pool_h
            row_t_bot = (pool_h - 1 - col_row) / pool_h
            for is_bot, row_t in [(False, row_t_top), (True, row_t_bot)]:
                y_row = tp_y1 + (pool_h - 1 - col_row if is_bot else col_row)
                x_left = int(tp_x1 + (tp_x1b - tp_x1) * row_t)
                x_right = int(tp_x2 + (tp_x2b - tp_x2) * row_t)
                if 0 <= y_row < GIF_H and x_left < x_right:
                    w_row = x_right - x_left
                    frame_arr[y_row, x_left:x_right] = np.resize(concrete[col_row, :pool_w], (w_row, 3))
        # Rebar grid overlay (visible at >50% progress)
        if t > 0.5:
            rebar_color = np.array([40, 35, 30], dtype=np.uint8)
            spacing = max(8, pool_h // 8)
            for row in range(pool_h):
                row_t = row / pool_h
                x_left = int(tp_x1 + (tp_x1b - tp_x1) * row_t)
                x_right = int(tp_x2 + (tp_x2b - tp_x2) * row_t)
                y_row = tp_y1 + row
                if 0 <= y_row < GIF_H and row % spacing == 0:
                    for px_r in range(max(0, x_left + wall), min(GIF_W, x_right - wall)):
                        frame_arr[y_row, px_r] = rebar_color
            spacing_c = max(8, pool_w // 8)
            for col_off in range(0, pool_w, spacing_c):
                for row in range(pool_h):
                    row_t = row / pool_h
                    x_left = int(tp_x1 + (tp_x1b - tp_x1) * row_t)
                    px_r = x_left + wall + int(col_off * (x_right - x_left - 2 * wall) / pool_w)
                    y_row = tp_y1 + row
                    if 0 <= y_row < GIF_H and 0 <= px_r < GIF_W:
                        frame_arr[y_row, px_r] = rebar_color
        add(frame_arr, 90)

    # ── Stage 6: Water rises (16 frames) ──────────────────────────────────────
    n_water = 16
    shimmer_offset = 0
    for i in range(n_water):
        t = (i + 1) / n_water
        frame_arr = tilted_arr.copy()
        wall = max(2, int(pool_w * 0.10))
        # Draw full concrete shell first
        for row in range(pool_h):
            row_t = row / pool_h
            x_left = int(tp_x1 + (tp_x1b - tp_x1) * row_t)
            x_right = int(tp_x2 + (tp_x2b - tp_x2) * row_t)
            y_row = tp_y1 + row
            if 0 <= y_row < GIF_H and x_left < x_right:
                w_row = x_right - x_left
                frame_arr[y_row, x_left:x_right] = np.resize(concrete[row, :pool_w], (w_row, 3))
        # Water fills from bottom
        water_rows = int(pool_h * t)
        water_start_row = pool_h - water_rows
        for row in range(water_start_row, pool_h):
            row_t = row / pool_h
            x_left = int(tp_x1 + (tp_x1b - tp_x1) * row_t) + wall
            x_right = int(tp_x2 + (tp_x2b - tp_x2) * row_t) - wall
            y_row = tp_y1 + row
            if 0 <= y_row < GIF_H and x_left < x_right:
                depth = (row - water_start_row) / max(1, water_rows)
                # Deep blue at bottom, lighter cyan at surface
                r = int(0 + depth * 10)
                g = int(80 + depth * 50)
                b = int(180 + depth * 40)
                water_color = np.array([r, g, b], dtype=np.uint8)
                frame_arr[y_row, x_left:x_right] = water_color
        # Shimmer line at water surface
        surface_row = tp_y1 + water_start_row
        if 0 <= surface_row < GIF_H:
            row_t = water_start_row / pool_h
            x_left = int(tp_x1 + (tp_x1b - tp_x1) * row_t) + wall
            x_right = int(tp_x2 + (tp_x2b - tp_x2) * row_t) - wall
            shimmer_col = np.array([200, 230, 255], dtype=np.uint8)
            for sx in range(x_left + shimmer_offset % 12, x_right, 20):
                for sw in range(min(8, x_right - sx)):
                    if 0 <= sx + sw < GIF_W:
                        frame_arr[surface_row, sx + sw] = shimmer_col
            shimmer_offset += 4
        add(frame_arr, 80)

    # ── Stage 7: Hard cut to AI render (hold 8 frames × 400ms) ───────────────
    if rendered:
        for _ in range(8):
            add(rendered, 400)
    else:
        # If no AI render, hold on completed pool state
        for _ in range(4):
            add(Image.fromarray(frame_arr), 400)

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
    logger.info(f"Cinematic GIF saved: {out_path} ({len(frames)} frames, {sum(durations)//1000}s)")
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
