"""
Postcard generator — 6x9 glossy BEFORE/AFTER layout.

Left half:  satellite aerial photo  (BEFORE)
Right half: AI-rendered pool photo  (AFTER)
Bottom bar: address · build cost · value lift · QR code · contractor

Output: JPEG at 1875x1275 (300 dpi) for Lob's 6x9 spec.
"""
import os
import asyncio
import logging
import qrcode
from PIL import Image, ImageDraw, ImageFont
import io
import config

logger = logging.getLogger(__name__)

# Canvas: 6x9 inches @ 300 dpi
W, H = 1875, 1275
HALF = W // 2          # 937
BAR_H = 220
IMG_H = H - BAR_H      # 1055

C_GREEN  = (34, 139, 34)
C_WHITE  = (255, 255, 255)
C_BLACK  = (20, 20, 20)
C_GRAY   = (120, 120, 120)
C_PANEL  = (245, 245, 240)
C_BADGE_BEFORE = (60, 60, 60, 210)
C_BADGE_AFTER  = (34, 139, 34, 210)


def _load_font(size: int, bold: bool = False):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


async def generate_postcard(
    prospect_id: str,
    address: str,
    city: str,
    state: str,
    rendered_image_path: str,
    satellite_image_path: str = "",
    pool_cost: float = 0,
    value_lift: float = 0,
    microsite_url: str = "",
    contractor_name: str = "Premier Pool & Spa",
    contractor_phone: str = "(813) 555-0190",
) -> str:
    out_path = os.path.join(config.IMAGES_DIR, f"{prospect_id}_postcard.jpg")
    await asyncio.sleep(0)

    canvas = Image.new("RGB", (W, H), C_WHITE)
    draw = ImageDraw.Draw(canvas, "RGBA")

    # Left panel: satellite (BEFORE)
    try:
        sat = Image.open(satellite_image_path).convert("RGB").resize((HALF, IMG_H), Image.LANCZOS)
    except Exception:
        sat = Image.new("RGB", (HALF, IMG_H), (80, 80, 80))
    canvas.paste(sat, (0, 0))

    # Right panel: rendered pool (AFTER)
    try:
        ren = Image.open(rendered_image_path).convert("RGB").resize((HALF, IMG_H), Image.LANCZOS)
    except Exception:
        ren = Image.new("RGB", (HALF, IMG_H), (30, 100, 180))
    canvas.paste(ren, (HALF, 0))

    # Centre divider
    draw.rectangle([(HALF - 3, 0), (HALF + 3, IMG_H)], fill=C_WHITE)

    # BEFORE / AFTER badges
    badge_font = _load_font(38, bold=True)
    for label, bx, color in [("BEFORE", 28, C_BADGE_BEFORE), ("AFTER", HALF + 28, C_BADGE_AFTER)]:
        bbox = draw.textbbox((0, 0), label, font=badge_font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad = 14
        draw.rounded_rectangle(
            [(bx - pad, 28 - pad), (bx + tw + pad, 28 + th + pad)],
            radius=8, fill=color,
        )
        draw.text((bx, 28), label, font=badge_font, fill=C_WHITE)

    # Bottom info bar
    bar_y = IMG_H
    draw.rectangle([(0, bar_y), (W, H)], fill=C_PANEL)
    draw.rectangle([(0, bar_y), (W, bar_y + 6)], fill=C_GREEN)

    addr_font  = _load_font(36, bold=True)
    city_font  = _load_font(28)
    label_font = _load_font(24)
    num_font   = _load_font(44, bold=True)
    small_font = _load_font(22)
    ctr_font   = _load_font(30, bold=True)

    addr_x, addr_y = 40, bar_y + 22
    draw.text((addr_x, addr_y), address, font=addr_font, fill=C_BLACK)
    draw.text((addr_x, addr_y + 44), f"{city}, {state}  ·  Pool ready", font=city_font, fill=C_GRAY)

    col1_x = 40
    col2_x = 360
    num_y  = addr_y + 100
    draw.text((col1_x, num_y), f"${int(pool_cost):,}", font=num_font, fill=C_GREEN)
    draw.text((col1_x, num_y + 52), "Estimated build cost", font=label_font, fill=C_GRAY)
    draw.text((col2_x, num_y), f"+${int(value_lift):,}", font=num_font, fill=C_BLACK)
    draw.text((col2_x, num_y + 52), "Projected value lift", font=label_font, fill=C_GRAY)

    ctr_x = W - 520
    draw.text((ctr_x, addr_y + 10), contractor_name, font=ctr_font, fill=C_BLACK)
    draw.text((ctr_x, addr_y + 48), contractor_phone, font=city_font, fill=C_GRAY)
    draw.text((ctr_x, addr_y + 82), "Free consultation · No obligation", font=small_font, fill=C_GRAY)

    # QR code
    qr_url = f"https://{microsite_url}" if not microsite_url.startswith("http") else microsite_url
    qr = qrcode.QRCode(version=1, box_size=4, border=2)
    qr.add_data(qr_url)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    qr_size = 160
    qr_img = qr_img.resize((qr_size, qr_size), Image.LANCZOS)
    qr_x = W - qr_size - 30
    qr_y = bar_y + (BAR_H - qr_size) // 2
    canvas.paste(qr_img, (qr_x, qr_y))
    draw.text((qr_x + 8, qr_y + qr_size + 4), "Scan to see your pool", font=small_font, fill=C_GRAY)

    # Legal footer
    legal = f"{contractor_name} · Licensed & Insured · FL #CPC1234567"
    legal_font = _load_font(18)
    draw.text((HALF - 200, H - 22), legal, font=legal_font, fill=C_GRAY)

    canvas.save(out_path, "JPEG", quality=92)
    logger.info(f"Postcard saved: {out_path}")
    return out_path
