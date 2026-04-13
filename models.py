import uuid
from datetime import datetime
from sqlalchemy import String, Float, Integer, Boolean, DateTime, Text, ForeignKey, Enum
from sqlalchemy.orm import Mapped, mapped_column, relationship
import enum

from database import Base


class PipelineStatus(str, enum.Enum):
    pending = "pending"
    running = "running"
    complete = "complete"
    error = "error"


class Campaign(Base):
    __tablename__ = "campaigns"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    target_region: Mapped[str] = mapped_column(String, default="")
    min_home_value: Mapped[float] = mapped_column(Float, default=500_000)
    max_home_value: Mapped[float] = mapped_column(Float, default=1_500_000)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    prospects: Mapped[list["Prospect"]] = relationship("Prospect", back_populates="campaign")


class Prospect(Base):
    __tablename__ = "prospects"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    campaign_id: Mapped[str] = mapped_column(String, ForeignKey("campaigns.id"), nullable=False)

    # Property info
    address: Mapped[str] = mapped_column(String, nullable=False)
    city: Mapped[str] = mapped_column(String, default="")
    state: Mapped[str] = mapped_column(String, default="")
    zip_code: Mapped[str] = mapped_column(String, default="")
    lat: Mapped[float] = mapped_column(Float, default=0.0)
    lng: Mapped[float] = mapped_column(Float, default=0.0)
    home_value: Mapped[float] = mapped_column(Float, default=0.0)
    lot_sqft: Mapped[int] = mapped_column(Integer, default=0)
    year_built: Mapped[int] = mapped_column(Integer, default=0)

    # Pipeline state
    status: Mapped[str] = mapped_column(String, default=PipelineStatus.pending)
    current_step: Mapped[str] = mapped_column(String, default="")

    # Step outputs
    satellite_image_path: Mapped[str] = mapped_column(String, default="")
    rendered_image_path: Mapped[str] = mapped_column(String, default="")
    pool_sqft: Mapped[int] = mapped_column(Integer, default=0)
    pool_cost: Mapped[float] = mapped_column(Float, default=0.0)
    value_lift: Mapped[float] = mapped_column(Float, default=0.0)
    listing_agent_name: Mapped[str] = mapped_column(String, default="")
    listing_agency: Mapped[str] = mapped_column(String, default="")
    postcard_path: Mapped[str] = mapped_column(String, default="")
    postcard_lob_id: Mapped[str] = mapped_column(String, default="")
    microsite_url: Mapped[str] = mapped_column(String, default="")
    microsite_viewed: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    campaign: Mapped["Campaign"] = relationship("Campaign", back_populates="prospects")
    events: Mapped[list["PipelineEvent"]] = relationship("PipelineEvent", back_populates="prospect", order_by="PipelineEvent.created_at.desc()")


class PipelineEvent(Base):
    __tablename__ = "pipeline_events"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    prospect_id: Mapped[str] = mapped_column(String, ForeignKey("prospects.id"), nullable=False)
    step: Mapped[str] = mapped_column(String, nullable=False)
    label: Mapped[str] = mapped_column(String, default="")
    detail: Mapped[str] = mapped_column(String, default="")
    status: Mapped[str] = mapped_column(String, default="complete")  # running | complete | error
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    prospect: Mapped["Prospect"] = relationship("Prospect", back_populates="events")


class Booking(Base):
    __tablename__ = "bookings"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    prospect_id: Mapped[str] = mapped_column(String, ForeignKey("prospects.id"), nullable=False)
    homeowner_name: Mapped[str] = mapped_column(String, default="")
    email: Mapped[str] = mapped_column(String, default="")
    phone: Mapped[str] = mapped_column(String, default="")
    appointment_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    notes: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
