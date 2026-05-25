"""Pydantic data models for the entire system.

This module contains only Pydantic v2 data models used by the Hub-and-Spoke
VM migration orchestrator (LAAS-CNRS). Models are immutable (frozen).
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RTTMeasurement(BaseModel):
    """RTT measurement from a Pi toward a VM in milliseconds."""

    vm_id: str
    rtt_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(frozen=True)


class LatencyPrediction(BaseModel):
    """Latency prediction series for a VM in milliseconds."""

    vm_id: str
    current_ms: float
    predicted: list[float]

    model_config = ConfigDict(frozen=True)


class VMMetrics(BaseModel):
    """Instantaneous VM resource utilization metrics."""

    vm_id: str
    cpu_percent: float
    ram_percent: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(frozen=True)


class CPUPrediction(BaseModel):
    """CPU usage prediction series for a VM (percentage)."""

    vm_id: str
    current_cpu: float
    predicted: list[float]

    model_config = ConfigDict(frozen=True)


class RAMPrediction(BaseModel):
    """RAM usage prediction series for a VM (percentage)."""

    vm_id: str
    current_ram: float
    predicted: list[float]

    model_config = ConfigDict(frozen=True)


class SLO(BaseModel):
    metric: Literal["latency", "cpu_usage", "ram_usage"]
    operator: Literal["<", "<=", ">", ">="]
    threshold: float
    unit: Literal["ms", "%"]


class VMFullPrediction(BaseModel):
    """Aggregated predictions (latency, cpu, ram) for a VM."""

    vm_id: str
    latency: LatencyPrediction
    cpu: CPUPrediction
    ram: RAMPrediction

    model_config = ConfigDict(frozen=True)


class DecisionResult(BaseModel):
    """Result of a migration decision for a VM pair or stay decision."""

    decision: Literal["migrate", "stay"]
    from_vm: str | None
    to_vm: str | None
    reason: str
    mode: Literal["classic", "enhanced"]

    model_config = ConfigDict(frozen=True)


class MasterCommand(BaseModel):
    """Command message issued by the orchestrator to effect a decision."""

    decision: str
    service: str
    from_vm: str | None
    to_vm: str | None
    mode: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    reason: str

    model_config = ConfigDict(frozen=True)


class IntentRequest(BaseModel):
    """Natural-language intent submitted to the Intent Engine."""

    intent_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    intention: str

    model_config = ConfigDict(frozen=True)


class SLOResponse(BaseModel):
    """Structured SLOs produced in response to an intent."""

    intent_id: str
    slos: list[SLO]

    model_config = ConfigDict(frozen=True)

