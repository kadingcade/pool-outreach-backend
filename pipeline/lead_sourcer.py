""" Lead sourcer — finds pool-ready homeowners.
Real mode: geocodes via Google Maps, uses realistic defaults for property data.
Demo mode: returns rich fake leads for any address entered.
"""
import random
import httpx
from config import DEMO_MODE, GOOGLE_MAPS_API_KEY

DEMO_AGENTS = [
    ("James Torres", "Keller Williams Realty"),
    ("Amanda Chen", "RE/MAX Premium"),
    ("David Park", "Coldwell Banker Residential"),
    ("Lisa Monroe", "CENTURY 21 Coastal Homes"),
    ("Robert Hughes", "eXp Realty"),
]

FLORIDA_CITIES = [
    ("Tampa", "FL", "33602", 27.9506, -82.4572),
    ("Wesley Chapel", "FL", "33543", 28.1869, -82.3468),
    ("Sarasota", "FL", "34236", 27.3364, -82.5307),
    ("Naples", "FL", "34102", 26.1420, -81.7948),
    ("Orlando", "FL", "32801", 28.5383, -81.3792),
    ("Clearwater", "FL", "33755", 27.9659, -82.8001),
    ("Fort Myers", "FL", "33901", 26.6406, -81.8723),
    ("Boca Raton", "FL", "33432", 26.3683, -80.1289),
]


async def enrich_address(address: str) -> dict:
    """
    Given a raw address string, return enriched property data.
    In demo mode, generates realistic data.
    In real mode, geocodes via Google Maps + realistic property defaults.
    """
    if DEMO_MODE or not GOOGLE_MAPS_API_KEY:
        return _demo_enrich(address)
    return await _real_enrich(address)


def _demo_enrich(address: str) -> dict:
    """Generate realistic property data for demo/testing."""
    city_data = random.choice(FLORIDA_CITIES)
    agent = random.choice(DEMO_AGENTS)
    parts = address.split(",")
    city = city_data[0]
    state = city_data[1]
    zip_code = city_data[2]
    lat = city_data[3] + random.uniform(-0.05, 0.05)
    lng = city_data[4] + random.uniform(-0.05, 0.05)
    if len(parts) >= 2:
        city_part = parts[-2].strip()
        if city_part:
            city = city_part
    return {
        "city": city,
        "state": state,
        "zip_code": zip_code,
        "lat": lat,
        "lng": lng,
        "home_value": random.randint(500, 1500) * 1000,
        "lot_sqft": random.randint(10_000, 35_000),
        "year_built": random.randint(2005, 2022),
        "listing_agent_name": random.choice(DEMO_AGENTS)[0],
        "listing_agency": random.choice(DEMO_AGENTS)[1],
    }


async def _real_enrich(address: str) -> dict:
    """Geocode via Google Maps. Use realistic defaults for property fields."""
    agent = random.choice(DEMO_AGENTS)
    async with httpx.AsyncClient(timeout=10) as client:
        geo_resp = await client.get(
            "https://maps.googleapis.com/maps/api/geocode/json",
            params={"address": address, "key": GOOGLE_MAPS_API_KEY},
        )
        geo = geo_resp.json()

        if not geo.get("results"):
            return _demo_enrich(address)

        result = geo["results"][0]
        loc = result["geometry"]["location"]
        components = {
            c["types"][0]: c["long_name"]
            for c in result["address_components"]
        }

        return {
            "city": components.get("locality", components.get("sublocality", "")),
            "state": components.get("administrative_area_level_1", ""),
            "zip_code": components.get("postal_code", ""),
            "lat": loc["lat"],
            "lng": loc["lng"],
            # Realistic defaults — would be replaced by BatchData/Attom in production
            "home_value": random.randint(500, 1500) * 1000,
            "lot_sqft": random.randint(10_000, 35_000),
            "year_built": random.randint(2005, 2022),
            "listing_agent_name": agent[0],
            "listing_agency": agent[1],
        }
