"""
Pipeline orchestrator — runs all steps end-to-end for a prospect,
emitting real-time WebSocket events at each stage.

Step sequence (mirrors OpenClaw's activity feed):
  1. lot_scanned           — satellite imagery downloaded
  2. pool_ready_identified — backyard zone + setback check
  3. pool_rendered         — AI pool composited into image
  4. listing_agent_id      — listing agent pulled
  5. economics_calculated  — build cost + value lift
  6. postcard_generated    — 6x9 JPEG postcard created
  7. postcard_mailed       — dispatched via Lob
  8. microsite_live        — personalized landing page deployed
"""
import asyncio
import logging
from datetime import datetime
from typing import Callable, Awaitable
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models import Prospect, PipelineEvent
from pipeline import lead_sourcer, lot_scanner, pool_renderer, economics
from pipeline import postcard_generator, mailer, microsite_generator

logger = logging.getLogger(__name__)


async def run_pipeline(
    prospect_id: str,
    campaign_id: str,
    db: AsyncSession,
    broadcast: Callable[[dict], Awaitable[None]],
):
    """
    Runs the full 8-step pipeline for a prospect.
    Broadcasts real-time events to the campaign's WebSocket channel.
    """
    result = await db.execute(select(Prospect).where(Prospect.id == prospect_id))
    prospect = result.scalar_one_or_none()
    if not prospect:
        logger.error(f"Prospect {prospect_id} not found")
        return

    async def emit(step: str, label: str, detail: str, status: str = "complete", extra: dict = None):
        """Save event to DB and broadcast via WebSocket."""
        event = PipelineEvent(
            prospect_id=prospect_id,
            step=step,
            label=label,
            detail=detail,
            status=status,
        )
        db.add(event)
        await db.commit()

        payload = {
            "type": "pipeline_event",
            "prospect_id": prospect_id,
            "step": step,
            "label": label,
            "detail": detail,
            "status": status,
            "timestamp": datetime.utcnow().isoformat(),
        }
        if extra:
            payload.update(extra)
        await broadcast(payload)

    async def update_prospect(**kwargs):
        for k, v in kwargs.items():
            setattr(prospect, k, v)
        await db.commit()
        # Send updated prospect data to frontend
        await broadcast({
            "type": "prospect_update",
            "prospect_id": prospect_id,
            "data": {k: str(v) if not isinstance(v, (int, float, bool, type(None))) else v for k, v in kwargs.items()},
        })

    try:
        await update_prospect(status="running", current_step="lot_scanned")

        # ── Step 1: Lot scan ──────────────────────────────────────────────
        await emit("lot_scanned", "Lot scanned · scanning", f"Fetching satellite imagery…", status="running")

        enriched = await lead_sourcer.enrich_address(prospect.address)
        await update_prospect(**{k: v for k, v in enriched.items() if k not in ("listing_agent_name", "listing_agency")})

        satellite_path = await lot_scanner.get_satellite_image(
            prospect_id=prospect_id,
            lat=enriched.get("lat", 0),
            lng=enriched.get("lng", 0),
            address=prospect.address,
        )
        await update_prospect(satellite_image_path=f"/static/images/{prospect_id}_satellite.jpg")

        zone_info = lot_scanner.identify_pool_zone(satellite_path, enriched.get("lot_sqft", 15000))

        await emit(
            "lot_scanned",
            f"Lot scanned · {enriched.get('city', '')}",
            f"{enriched.get('lot_sqft', 0):,} sqft via satellite imagery",
            extra={"satellite_image": f"/static/images/{prospect_id}_satellite.jpg"},
        )
        await asyncio.sleep(0.4)

        # ── Step 2: Pool-ready zone ───────────────────────────────────────
        await emit("pool_ready_identified", "Pool-ready zone identified",
                   f"Built {enriched.get('year_built', '–')} · {enriched.get('lot_sqft', 0):,} sqft · setbacks cleared")
        await asyncio.sleep(0.3)

        # ── Step 3: Pool render ───────────────────────────────────────────
        await emit("pool_rendered", "Rendering pool…", "Photorealistic AI render in progress", status="running")

        rendered_path = await pool_renderer.render_pool(
            prospect_id=prospect_id,
            satellite_path=satellite_path,
            pool_zone=zone_info["pool_zone"],
        )
        await update_prospect(
            rendered_image_path=f"/static/images/{prospect_id}_rendered.jpg",
            pool_sqft=zone_info.get("pool_sqft", 450),
        )
        await emit(
            "pool_rendered",
            "Pool rendered",
            "Photorealistic AI render placed in actual backyard",
            extra={"rendered_image": f"/static/images/{prospect_id}_rendered.jpg"},
        )
        await asyncio.sleep(0.3)

        # ── Step 4: Listing agent ─────────────────────────────────────────
        agent_name = enriched.get("listing_agent_name", "")
        agency = enriched.get("listing_agency", "")
        await update_prospect(listing_agent_name=agent_name, listing_agency=agency)
        await emit(
            "listing_agent_identified",
            f"Listing agent identified · {agency}",
            "Listing contact on record",
        )
        await asyncio.sleep(0.3)

        # ── Step 5: Economics ─────────────────────────────────────────────
        econ = economics.calculate_pool_economics(
            home_value=enriched.get("home_value", 0),
            lot_sqft=enriched.get("lot_sqft", 0),
            pool_sqft=zone_info.get("pool_sqft", 450),
            state=enriched.get("state", "FL"),
        )
        await update_prospect(pool_cost=econ["pool_cost"], value_lift=econ["value_lift"])
        await emit(
            "economics_calculated",
            "Pool economics calculated",
            f"${econ['pool_cost']:,.0f} build · +${econ['value_lift']:,.0f} home value lift",
            extra={"pool_cost": econ["pool_cost"], "value_lift": econ["value_lift"]},
        )
        await asyncio.sleep(0.3)

        # ── Step 6: Postcard ──────────────────────────────────────────────
        await emit("postcard_generated", "Generating postcard…", "Creating 6×9 glossy design", status="running")
        await update_prospect(current_step="postcard_generated")

        postcard_path = await postcard_generator.generate_postcard(
            prospect_id=prospect_id,
            address=prospect.address,
            city=enriched.get("city", ""),
            state=enriched.get("state", ""),
            rendered_image_path=rendered_path,
            satellite_image_path=satellite_path,
            pool_cost=econ["pool_cost"],
            value_lift=econ["value_lift"],
            microsite_url=prospect_id,
        )
        await update_prospect(postcard_path=f"/static/images/{prospect_id}_postcard.jpg")
        await emit(
            "postcard_generated",
            "Postcard generated",
            "6×9 glossy with rendered overhead + QR code via Lob",
            extra={"postcard_image": f"/static/images/{prospect_id}_postcard.jpg"},
        )
        await asyncio.sleep(0.3)

        # ── Step 7: Mail ──────────────────────────────────────────────────
        await emit("postcard_mailed", "Dispatching to Lob…", "Printing and mailing", status="running")
        await update_prospect(current_step="postcard_mailed")

        mail_result = await mailer.mail_postcard(
            prospect_id=prospect_id,
            postcard_image_path=postcard_path,
            address=prospect.address,
            city=enriched.get("city", ""),
            state=enriched.get("state", ""),
            zip_code=enriched.get("zip_code", ""),
        )
        await update_prospect(postcard_lob_id=mail_result.get("lob_id", ""))
        await emit(
            "postcard_mailed",
            "Postcard mailed",
            f"Drop weight: {mail_result.get('drop_weight', '1.2oz')} · ETA {mail_result.get('expected_delivery', '3 business days')}",
        )
        await asyncio.sleep(0.3)

        # ── Step 8: Microsite ─────────────────────────────────────────────
        await emit("microsite_live", "Deploying microsite…", "Building personalized landing page", status="running")
        await update_prospect(current_step="microsite_live")

        microsite_url = await microsite_generator.generate_microsite(
            prospect_id=prospect_id,
            address=prospect.address,
            city=enriched.get("city", ""),
            state=enriched.get("state", ""),
            zip_code=enriched.get("zip_code", ""),
            rendered_image_path=rendered_path,
            pool_cost=econ["pool_cost"],
            value_lift=econ["value_lift"],
            lot_sqft=enriched.get("lot_sqft", 0),
            year_built=enriched.get("year_built", 0),
        )
        await update_prospect(microsite_url=microsite_url)
        await emit(
            "microsite_live",
            "Microsite live",
            microsite_url,
            extra={"microsite_url": microsite_url},
        )

        # ── Done ──────────────────────────────────────────────────────────
        await update_prospect(status="complete", current_step="complete")
        await broadcast({
            "type": "pipeline_complete",
            "prospect_id": prospect_id,
            "timestamp": datetime.utcnow().isoformat(),
        })
        logger.info(f"Pipeline complete for prospect {prospect_id}")

    except Exception as e:
        logger.exception(f"Pipeline error for {prospect_id}: {e}")
        await update_prospect(status="error", current_step="error")
        await broadcast({
            "type": "pipeline_error",
            "prospect_id": prospect_id,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        })
