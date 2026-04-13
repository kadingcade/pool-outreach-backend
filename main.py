import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

import config
from database import init_db, get_db
from models import Campaign, Prospect, PipelineEvent, Booking
from pipeline.orchestrator import run_pipeline


# ── WebSocket connection manager ──────────────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self.active: dict[str, list[WebSocket]] = {}  # campaign_id -> [ws]

    async def connect(self, ws: WebSocket, campaign_id: str):
        await ws.accept()
        self.active.setdefault(campaign_id, []).append(ws)

    def disconnect(self, ws: WebSocket, campaign_id: str):
        self.active.get(campaign_id, []).remove(ws)

    async def broadcast(self, campaign_id: str, data: dict):
        dead = []
        for ws in self.active.get(campaign_id, []):
            try:
                await ws.send_text(json.dumps(data, default=str))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.active.get(campaign_id, []).remove(ws)


manager = ConnectionManager()


# ── App lifecycle ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    await seed_demo_data()
    yield


app = FastAPI(title="Pool Outreach", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=config.STATIC_DIR, html=True), name="static")


# ── Demo seed data ────────────────────────────────────────────────────────────

async def seed_demo_data():
    from database import AsyncSessionLocal
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Campaign))
        if result.scalars().first():
            return  # already seeded

        campaign = Campaign(
            id="demo-campaign-1",
            name="Pool – Sun Belt Mid-Market",
            description="Targeting pool-ready homes $500K–$1.5M across Sun Belt metros",
            target_region="Tampa, FL",
            min_home_value=500_000,
            max_home_value=1_500_000,
        )
        db.add(campaign)
        await db.commit()


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class CampaignOut(BaseModel):
    id: str
    name: str
    description: str
    target_region: str
    min_home_value: float
    max_home_value: float
    created_at: datetime
    is_active: bool
    model_config = {"from_attributes": True}


class ProspectOut(BaseModel):
    id: str
    campaign_id: str
    address: str
    city: str
    state: str
    zip_code: str
    home_value: float
    lot_sqft: int
    year_built: int
    status: str
    current_step: str
    satellite_image_path: str
    rendered_image_path: str
    pool_cost: float
    value_lift: float
    listing_agent_name: str
    listing_agency: str
    postcard_path: str
    microsite_url: str
    microsite_viewed: bool
    created_at: datetime
    model_config = {"from_attributes": True}


class EventOut(BaseModel):
    id: str
    prospect_id: str
    step: str
    label: str
    detail: str
    status: str
    created_at: datetime
    model_config = {"from_attributes": True}


class RunPipelineRequest(BaseModel):
    campaign_id: str
    addresses: list[str]


class BookingCreate(BaseModel):
    prospect_id: str
    homeowner_name: str
    email: str
    phone: str
    appointment_at: datetime | None = None
    notes: str = ""


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/campaigns", response_model=list[CampaignOut])
async def list_campaigns(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Campaign).order_by(Campaign.created_at.desc()))
    return result.scalars().all()


@app.get("/campaigns/{campaign_id}", response_model=CampaignOut)
async def get_campaign(campaign_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Campaign).where(Campaign.id == campaign_id))
    campaign = result.scalar_one_or_none()
    if not campaign:
        raise HTTPException(404, "Campaign not found")
    return campaign


@app.get("/campaigns/{campaign_id}/prospects", response_model=list[ProspectOut])
async def list_prospects(campaign_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Prospect)
        .where(Prospect.campaign_id == campaign_id)
        .order_by(Prospect.created_at.desc())
    )
    return result.scalars().all()


@app.get("/prospects/{prospect_id}", response_model=ProspectOut)
async def get_prospect(prospect_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Prospect).where(Prospect.id == prospect_id))
    p = result.scalar_one_or_none()
    if not p:
        raise HTTPException(404, "Prospect not found")
    return p


@app.get("/prospects/{prospect_id}/events", response_model=list[EventOut])
async def get_events(prospect_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(PipelineEvent)
        .where(PipelineEvent.prospect_id == prospect_id)
        .order_by(PipelineEvent.created_at.desc())
    )
    return result.scalars().all()


@app.post("/pipeline/run")
async def trigger_pipeline(
    req: RunPipelineRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Campaign).where(Campaign.id == req.campaign_id))
    campaign = result.scalar_one_or_none()
    if not campaign:
        raise HTTPException(404, "Campaign not found")

    prospect_ids = []
    for address in req.addresses:
        p = Prospect(
            id=str(uuid.uuid4()),
            campaign_id=req.campaign_id,
            address=address,
            status="pending",
        )
        db.add(p)
        prospect_ids.append(p.id)
    await db.commit()

    # Fire pipeline for each prospect as a background task
    for pid in prospect_ids:
        background_tasks.add_task(_run_pipeline_task, pid, req.campaign_id)

    return {"started": prospect_ids}


async def _run_pipeline_task(prospect_id: str, campaign_id: str):
    from database import AsyncSessionLocal
    async with AsyncSessionLocal() as db:
        await run_pipeline(
            prospect_id=prospect_id,
            campaign_id=campaign_id,
            db=db,
            broadcast=lambda data: manager.broadcast(campaign_id, data),
        )


@app.post("/bookings", response_model=dict)
async def create_booking(body: BookingCreate, db: AsyncSession = Depends(get_db)):
    booking = Booking(**body.model_dump())
    db.add(booking)
    await db.commit()
    return {"id": booking.id}


@app.get("/campaigns/{campaign_id}/bookings")
async def list_bookings(campaign_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Booking)
        .join(Prospect, Booking.prospect_id == Prospect.id)
        .where(Prospect.campaign_id == campaign_id)
        .order_by(Booking.created_at.desc())
    )
    bookings = result.scalars().all()
    return [
        {
            "id": b.id,
            "prospect_id": b.prospect_id,
            "homeowner_name": b.homeowner_name,
            "email": b.email,
            "phone": b.phone,
            "appointment_at": b.appointment_at,
            "notes": b.notes,
            "created_at": b.created_at,
        }
        for b in bookings
    ]


# Microsite QR scan tracking
@app.get("/properties/{prospect_id}")
async def microsite_view(prospect_id: str, db: AsyncSession = Depends(get_db)):
    from fastapi.responses import HTMLResponse
    import os
    result = await db.execute(select(Prospect).where(Prospect.id == prospect_id))
    p = result.scalar_one_or_none()
    if p:
        p.microsite_viewed = True
        await db.commit()
        # Try to serve generated microsite HTML
        microsite_path = os.path.join(config.MICROSITES_DIR, f"{prospect_id}.html")
        if os.path.exists(microsite_path):
            with open(microsite_path) as f:
                return HTMLResponse(f.read())
    raise HTTPException(404, "Property not found")


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws/{campaign_id}")
async def websocket_endpoint(ws: WebSocket, campaign_id: str):
    await manager.connect(ws, campaign_id)
    try:
        while True:
            await ws.receive_text()  # keep alive
    except WebSocketDisconnect:
        manager.disconnect(ws, campaign_id)
