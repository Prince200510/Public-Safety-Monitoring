from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

class RiskLevel(str, Enum):
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

@dataclass
class Alert:
    id: str
    created_at: datetime
    user_email: str
    location: str
    risk_level: RiskLevel
    risk_score: float
    file_name: str
    event_time_seconds: float
    confidence: float = 0.0
    primary_cause: str = ""
    supporting_factors: Optional[list[str]] = None
    explanation: str = ""
    acknowledged_at: Optional[datetime] = None
    emergency_response: Optional[dict] = None  # Gemini AI recommendation


@dataclass
class EmergencyResponse:
    """Emergency response recommendation for HIGH-risk events"""
    time: str
    risk_level: str
    unit: str  # Type of response unit
    action: str  # Primary action to take
    urgency: str  # Immediate/High/Medium
    reasoning: list[str]  # List of reasoning points

@dataclass
class UserLocation:
    user_email: str
    latitude: float
    longitude: float
    timestamp: datetime
    active: bool = True
