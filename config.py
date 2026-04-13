import os
from dotenv import load_dotenv

load_dotenv()

# --- Google Maps ---
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")

# --- AI rendering — pick one provider ---
# RENDER_PROVIDER options: "fal" | "openai" | "stability" | "replicate"
RENDER_PROVIDER = os.getenv("RENDER_PROVIDER", "fal")

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN", "")
FAL_API_KEY        = os.getenv("FAL_API_KEY", "")          # https://fal.ai/dashboard/keys
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY", "")       # https://platform.openai.com/api-keys
STABILITY_API_KEY  = os.getenv("STABILITY_API_KEY", "")    # https://platform.stability.ai/account/keys

# --- Lob (direct mail) ---
LOB_API_KEY = os.getenv("LOB_API_KEY", "")

# --- App settings ---
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
MICROSITE_BASE_DOMAIN = os.getenv("MICROSITE_BASE_DOMAIN", "localhost:8000/properties")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:////tmp/pool_outreach.db")

# --- Storage ---
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
IMAGES_DIR = os.path.join(STATIC_DIR, "images")
MICROSITES_DIR = os.path.join(STATIC_DIR, "microsites")

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(MICROSITES_DIR, exist_ok=True)

# --- Demo mode ---
# If True, uses fake/mock data so you can run without any API keys
DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() == "true"

# --- Pool cost model ---
POOL_BASE_COST = 45000          # base cost in USD
POOL_COST_PER_SQFT = 50        # additional cost per backyard sqft over threshold
HOME_VALUE_LIFT_PCT = 0.07     # pools typically add ~7% to home value

# --- Lead targeting ---
MIN_HOME_VALUE = 500_000
MAX_HOME_VALUE = 1_500_000
MIN_LOT_SQFT = 8_000
