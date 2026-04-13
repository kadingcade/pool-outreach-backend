"""
Postcard generator — creates a 6x9 glossy postcard with:
  - AI-rendered aerial pool image
  - Property address + economics callouts
  - Personalized QR code linking to microsite
  - Contractor branding

Output: JPEG suitable for Lob's 6x9 postcard spec (1875x1275 at 300dpi).
"""
import os
import asyncio
import qrcode
from PIL import Image, ImageDraw, ImageFont
import io

import config

# Postcard size: 6x9 inches at 300 DPI
POSTCARD_W = 1875  # 6.25" x 300
POSTCARD_H = 1275  # 4.25" x 300


def _load_font(size: int, bold: bool = False):
    """Load a font, falling back to PIL default if not available."""
    try:
        if bold:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


async def generate_postcard(
    prospect_id: str,
    address: str,
    city: str,
    state: str,
    rendered_image_path: str,
    pool_cost: float,
    value_lift: float,
    microsite_url: str,
    contractor_name: str = "Premier Pool & Spa",
    contractor_phone: str = "(813) 555-0190",
) -> str:
    """Generate a postcard image. Returns local file path."""
    await asyncio.sleep(0.3)

    out_path = os.path.join(config.IMAGES_DIR, f"{prospect_id}_postcard.jpg")

    card = Image.new("RGB", (POSTCARD_W, POSTCARD_H), (255, 255, 255))
    draw = ImageDraw.Draw(card)

    # ── Left panel: rendered aerial image (60% width) ──────────────────────
    panel_w = int(POSTCARD_W * 0.62)
    try:
        aerial = Image.open(rendered_image_path).convert("RGB")
        aerial = aerial.resize((panel_w, POSTCARD_H), Image.LANCZOS)
        card.paste(aerial, (0, 0))
    except Exception:
        draw.rectangle([0, 0, panel_w, POSTCARD_H], fill=(30, 80, 50))

    # Gradient overlay on left panel for text legibility
    overlay = Image.new("RGBA", (panel_w, POSTCARD_H), (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    for y in range(POSTCARD_H // 2, POSTCARD_H):
        alpha = int(180 * (y - POSTCARD_H // 2) / (POSTCARD_H // 2))
        overlay_draw.line([(0, y), (panel_w, y)], fill=(0, 0, 0, alpha))
    card.paste(overlay, (0, 0), mask=overlay.split()[3])

    # Address overlay on image
    try:
        card_alpha = card.convert("RGBA")
        ov = Image.new("RGBA", card.size, (0, 0, 0, 0))
        ov_draw = ImageDraw.Draw(ov)
        for y in range(POSTCARD_H - 220, POSTCARD_H):
            alpha = int(160 * (y - (POSTCARD_H - 220)) / 220)
            ov_draw.line([(0, y), (panel_w, y)], fill=(0, 0, 0, alpha))
        card = Image.alpha_composite(card_alpha, ov).convert("RGB")
        draw = ImageDraw.Draw(card)
    except Exception:
        pass

    font_addr = _load_font(38, bold=True)
    font_sub = _load_font(28)
    draw.text((30, POSTCARD_H - 190), address, font=font_addr, fill=(255, 255, 255))
    draw.text((30, POSTCARD_H - 140), f"{city}, {state}", font=font_sub, fill=(220, 220, 220))

    # ── Right panel: offer + QR ─────────────────────────────────────────────
    rx = panel_w + 20
    rw = POSTCARD_W - panel_w - 40

    # Header
    font_headline = _load_font(52, bold=True)
    font_body = _load_font(32)
    font_small = _load_font(26)
    font_cta = _load_font(36, bold=True)
    font_price = _load_font(56, bold=True)
    font_label = _load_font(24)

    # Brand accent bar
    draw.rectangle([panel_w, 0, POSTCARD_W, 8], fill=(0, 160, 100))

    # Headline
    draw.text((rx, 40), "Your home is", font=font_body, fill=(80, 80, 80))
    draw.text((rx, 80), "pool-ready.", font=font_headline, fill=(0, 140, 80))

    # Divider
    draw.line([(rx, 158), (POSTCARD_W - 20, 158)], fill=(220, 220, 220), width=2)

    # Economics
    draw.text((rx, 175), "ESTIMATED BUILD COST", font=font_label, fill=(120, 120, 120))
    draw.text((rx, 205), f"${pool_cost:,.0f}", font=font_price, fill=(30, 30, 30))

    draw.text((rx, 290), "PROJECTED VALUE LIFT", font=font_label, fill=(120, 120, 120))
    draw.text((rx, 320), f"+${value_lift:,.0f}", font=font_price, fill=(0, 140, 80))

    draw.line([(rx, 405), (POSTCARD_W - 20, 405)], fill=(220, 220, 220), width=2)

    # CTA
    draw.text((rx, 425), "See your pool →", font=font_cta, fill=(0, 100, 200))
    draw.text((rx, 470), "Scan to view your", font=font_small, fill=(100, 100, 100))
    draw.text((rx, 498), "personalized render", font=font_small, fill=(100, 100, 100))

    # QR code
    qr = qrcode.QRCode(version=2, box_size=6, border=2)
    qr.add_data(f"http://{config.BASE_URL}/properties/{microsite_url}")
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    qr_size = 220
    qr_img = qr_img.resize((qr_size, qr_size), Image.NEAREST)
    qr_x = rx
    qr_y = 540
    card.paste(qr_img, (qr_x, qr_y))

    # Contractor info
    draw.line([(rx, 790), (POSTCARD_W - 20, 790)], fill=(220, 220, 220), width=1)
    draw.text((rx, 808), contractor_name, font=_load_font(30, bold=True), fill=(40, 40, 40))
    draw.text((rx, 848), contractor_phone, font=font_small, fill=(80, 80, 80))
    draw.text((rx, 880), "Free consultation • No obligation", font=font_small, fill=(120, 120, 120))

    # License bar
    draw.rectangle([0, POSTCARD_H - 35, POSTCARD_W, POSTCARD_H], fill=(245, 245, 245))
    draw.text((20, POSTCARD_H - 28), "Premier Pool & Spa · Licensed & Insured · FL #CPC1234567", font=_load_font(18), fill=(160, 160, 160))

    card.save(out_path, "JPEG", quality=95, dpi=(300, 300))
    return out_path
