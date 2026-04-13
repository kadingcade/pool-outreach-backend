"""
Microsite generator — creates a personalized landing page per property.

Each property gets a unique URL. When the homeowner scans the QR code
on their postcard, they land here. We track the visit.
"""
import os
import asyncio
import re
from jinja2 import Template

import config

MICROSITE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Your Pool Vision — {{ address }}</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0a0e14; color: #fff; }
    .hero { position: relative; height: 60vh; min-height: 380px; overflow: hidden; }
    .hero img { width: 100%; height: 100%; object-fit: cover; }
    .hero-overlay {
      position: absolute; inset: 0;
      background: linear-gradient(to bottom, rgba(0,0,0,0.1), rgba(0,0,0,0.65));
      display: flex; flex-direction: column; justify-content: flex-end; padding: 40px 24px;
    }
    .badge { display: inline-block; background: #00c470; color: #fff; font-size: 12px;
             font-weight: 700; padding: 4px 10px; border-radius: 20px; letter-spacing: 1px;
             margin-bottom: 12px; text-transform: uppercase; }
    .hero h1 { font-size: clamp(24px, 5vw, 42px); font-weight: 800; line-height: 1.1; }
    .hero p { margin-top: 8px; color: #ccc; font-size: 16px; }
    .stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1px; background: #1a2030; }
    .stat { background: #111820; padding: 20px 16px; text-align: center; }
    .stat .val { font-size: 28px; font-weight: 800; color: #00c470; }
    .stat .lbl { font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }
    .section { padding: 32px 24px; max-width: 600px; margin: 0 auto; }
    .section h2 { font-size: 22px; font-weight: 700; margin-bottom: 12px; }
    .section p { color: #aaa; line-height: 1.6; font-size: 15px; }
    .cta { background: #00c470; color: #fff; border: none; width: 100%; padding: 18px;
           font-size: 18px; font-weight: 700; border-radius: 10px; cursor: pointer; margin-top: 24px; }
    .cta:hover { background: #00a85c; }
    .cta-section { padding: 0 24px 48px; max-width: 600px; margin: 0 auto; }
    .form-group { margin-bottom: 14px; }
    .form-group input, .form-group textarea {
      width: 100%; padding: 14px; background: #1a2030; border: 1px solid #2a3040;
      border-radius: 8px; color: #fff; font-size: 15px;
    }
    .form-group input::placeholder, .form-group textarea::placeholder { color: #555; }
    .footer { background: #0a0e14; border-top: 1px solid #1a2030; padding: 24px; text-align: center; color: #444; font-size: 12px; }
    .tag { display: inline-block; background: #1a2030; color: #888; padding: 4px 10px; border-radius: 4px; font-size: 12px; margin: 4px 2px; }
  </style>
</head>
<body>
  <div class="hero">
    <img src="/static/images/{{ rendered_image_filename }}" alt="Your pool render" onerror="this.style.display='none'"/>
    <div class="hero-overlay">
      <span class="badge">✓ Pool-Ready Property</span>
      <h1>Your backyard<br/>could look like this.</h1>
      <p>{{ address }}, {{ city }}, {{ state }}</p>
    </div>
  </div>

  <div class="stats">
    <div class="stat">
      <div class="val">${{ pool_cost_fmt }}</div>
      <div class="lbl">Build Cost</div>
    </div>
    <div class="stat">
      <div class="val">+${{ value_lift_fmt }}</div>
      <div class="lbl">Home Value Lift</div>
    </div>
    <div class="stat">
      <div class="val">{{ lot_sqft_fmt }} sqft</div>
      <div class="lbl">Your Lot</div>
    </div>
  </div>

  <div class="section">
    <h2>Why now?</h2>
    <p>
      Your property at {{ address }} has been identified as pool-ready — you have the lot size,
      setback clearance, and home value that makes a pool an exceptional investment.
      Pools in {{ city }} add an average of <strong style="color:#00c470">${{ value_lift_fmt }}</strong> to home value
      and are enjoyed year-round in {{ state }}.
    </p>
    <br/>
    <p>
      <span class="tag">24,965 sqft lot</span>
      <span class="tag">Built {{ year_built }}</span>
      <span class="tag">Setbacks cleared</span>
      <span class="tag">No HOA restrictions found</span>
    </p>
  </div>

  <div class="cta-section">
    <h2 style="margin-bottom:16px; font-size:22px;">Book your free consultation</h2>
    <p style="color:#888; margin-bottom:20px; font-size:14px;">
      No obligation. We'll walk your yard, answer questions, and leave you with a full quote.
    </p>
    <form onsubmit="handleSubmit(event)">
      <div class="form-group">
        <input type="text" id="name" placeholder="Your name" required />
      </div>
      <div class="form-group">
        <input type="tel" id="phone" placeholder="Phone number" required />
      </div>
      <div class="form-group">
        <input type="email" id="email" placeholder="Email address" />
      </div>
      <div class="form-group">
        <textarea id="notes" rows="3" placeholder="Any questions or notes?"></textarea>
      </div>
      <button type="submit" class="cta">Request Free Consultation →</button>
    </form>
  </div>

  <div class="footer">
    Premier Pool &amp; Spa · Licensed &amp; Insured · FL #CPC1234567<br/>
    © 2026 · This offer was generated specifically for {{ address }}
  </div>

  <script>
    async function handleSubmit(e) {
      e.preventDefault();
      const btn = e.target.querySelector('button');
      btn.textContent = 'Sending...';
      btn.disabled = true;
      try {
        await fetch('/bookings', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            prospect_id: '{{ prospect_id }}',
            homeowner_name: document.getElementById('name').value,
            phone: document.getElementById('phone').value,
            email: document.getElementById('email').value,
            notes: document.getElementById('notes').value,
          })
        });
        btn.textContent = '✓ Request received! We\\'ll be in touch within 24 hours.';
        btn.style.background = '#1a2030';
      } catch(err) {
        btn.textContent = 'Try again';
        btn.disabled = false;
      }
    }
  </script>
</body>
</html>"""


def _make_slug(prospect_id: str, address: str, city: str) -> str:
    """Create a URL-friendly slug."""
    clean = re.sub(r'[^a-z0-9]+', '-', f"{city}-{address}".lower()).strip('-')
    short_id = prospect_id.replace('-', '')[:10]
    return f"{clean}-{short_id}"


async def generate_microsite(
    prospect_id: str,
    address: str,
    city: str,
    state: str,
    zip_code: str,
    rendered_image_path: str,
    pool_cost: float,
    value_lift: float,
    lot_sqft: int,
    year_built: int,
) -> str:
    """Generate personalized HTML microsite. Returns the prospect_id (used as URL key)."""
    await asyncio.sleep(0.2)

    rendered_filename = os.path.basename(rendered_image_path) if rendered_image_path else ""

    tmpl = Template(MICROSITE_TEMPLATE)
    html = tmpl.render(
        prospect_id=prospect_id,
        address=address,
        city=city,
        state=state,
        zip_code=zip_code,
        rendered_image_filename=rendered_filename,
        pool_cost_fmt=f"{pool_cost:,.0f}",
        value_lift_fmt=f"{value_lift:,.0f}",
        lot_sqft_fmt=f"{lot_sqft:,}",
        year_built=year_built,
    )

    out_path = os.path.join(config.MICROSITES_DIR, f"{prospect_id}.html")
    with open(out_path, "w") as f:
        f.write(html)

    # Return the URL for this microsite
    microsite_url = f"{config.MICROSITE_BASE_DOMAIN}/{prospect_id}"
    return microsite_url
