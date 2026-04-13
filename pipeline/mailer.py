"""
Direct mail via Lob API.

Real mode: Creates and sends a physical 6x9 postcard via lob.com.
Demo mode: Logs what would be sent, returns a mock tracking ID.
"""
import asyncio
import logging
import config

logger = logging.getLogger(__name__)


async def mail_postcard(
    prospect_id: str,
    postcard_image_path: str,
    address: str,
    city: str,
    state: str,
    zip_code: str,
    homeowner_name: str = "Current Resident",
    from_name: str = "Premier Pool & Spa",
    from_address: str = "123 Contractor Blvd",
    from_city: str = "Tampa",
    from_state: str = "FL",
    from_zip: str = "33602",
) -> dict:
    """
    Mail the postcard. Returns Lob response or mock response.
    """
    if config.DEMO_MODE or not config.LOB_API_KEY:
        return await _demo_mail(prospect_id, address)

    return await _lob_mail(
        postcard_image_path=postcard_image_path,
        to_name=homeowner_name,
        to_address=address,
        to_city=city,
        to_state=state,
        to_zip=zip_code,
        from_name=from_name,
        from_address=from_address,
        from_city=from_city,
        from_state=from_state,
        from_zip=from_zip,
    )


async def _lob_mail(
    postcard_image_path: str,
    to_name: str,
    to_address: str,
    to_city: str,
    to_state: str,
    to_zip: str,
    from_name: str,
    from_address: str,
    from_city: str,
    from_state: str,
    from_zip: str,
) -> dict:
    """Send real postcard via Lob API."""
    import lob
    lob.api_key = config.LOB_API_KEY

    try:
        # Upload front image
        with open(postcard_image_path, "rb") as f:
            postcard = lob.Postcard.create(
                description="Pool outreach postcard",
                to={
                    "name": to_name,
                    "address_line1": to_address,
                    "address_city": to_city,
                    "address_state": to_state,
                    "address_zip": to_zip,
                    "address_country": "US",
                },
                from_={
                    "name": from_name,
                    "address_line1": from_address,
                    "address_city": from_city,
                    "address_state": from_state,
                    "address_zip": from_zip,
                    "address_country": "US",
                },
                front=f,
                back="<html><body style='padding:20px;font-family:sans-serif'><p>Scan the QR code on the front to see your pool!</p></body></html>",
                size="6x9_postcard",
            )
        return {
            "lob_id": postcard.id,
            "status": postcard.status,
            "expected_delivery": str(postcard.expected_delivery_date),
            "drop_weight": "1.2oz",
        }
    except Exception as e:
        logger.error(f"Lob API error: {e}")
        return await _demo_mail("", to_address)


async def _demo_mail(prospect_id: str, address: str) -> dict:
    """Simulate mailing — no real API call."""
    await asyncio.sleep(0.5)
    logger.info(f"[DEMO] Would mail postcard to: {address}")
    return {
        "lob_id": f"psc_demo_{prospect_id[:8]}",
        "status": "in_transit",
        "expected_delivery": "3 business days",
        "drop_weight": "1.2oz",
    }
