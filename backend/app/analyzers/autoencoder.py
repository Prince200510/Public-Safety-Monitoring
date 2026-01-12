from __future__ import annotations

import os
from dataclasses import dataclass
from threading import Lock
from typing import List, Optional
import numpy as np
from ..models import RiskLevel
from ..path_setup import ensure_workspace_on_path

@dataclass
class AnalysisResult:
    risk_level: RiskLevel
    risk_score: float
    max_loss: float
    mean_loss: float
    event_time_seconds: float
    samples: Optional[List[dict]] = None
    losses: Optional[List[float]] = None


_model_lock = Lock()
_model = None

def _get_default_model_path() -> str:
    root_dir = ensure_workspace_on_path()
    return os.path.join(root_dir, "Crowd_Anomaly_Detection", "AnomalyDetector.h5")

def analyze_video_autoencoder(
    *,
    video_path: str,
    sample_every_seconds: float = 0.2,
    threshold_low: float = 0.0008,
    threshold_medium: float = 0.0012,
    threshold_high: float = 0.0016,
    include_losses: bool = False,
    stop_on_high: bool = True,
    model_path: Optional[str] = None,
) -> AnalysisResult:
    global _model
    ensure_workspace_on_path()

    from Crowd_Anomaly_Detection.run_video_risk_alerts import (
        _classify_risk,
        _extract_sampled_grayscale_frames,
        _load_model,
        _mean_euclidean_loss,
        _preprocess_to_model_tensor,
    )

    if model_path is None:
        model_path = _get_default_model_path()

    frames_gray = _extract_sampled_grayscale_frames(video_path, sample_every_seconds)
    bunches, _usable_frames = _preprocess_to_model_tensor(frames_gray)

    with _model_lock:
        if _model is None:
            _model = _load_model(model_path)
        model = _model

    losses: List[float] = []
    first_alert_bunch_idx: Optional[int] = None
    samples: List[dict] = []
    for bunch in bunches:
        n_bunch = np.expand_dims(bunch, axis=0)
        reconstructed = model.predict(n_bunch, verbose=0)
        loss = _mean_euclidean_loss(n_bunch, reconstructed)
        losses.append(loss)

        bunch_idx = len(losses) - 1
        seconds_per_bunch = 10.0 * float(sample_every_seconds)
        t_sec = float(bunch_idx * seconds_per_bunch)
        risk_str = _classify_risk(loss, threshold_low, threshold_medium, threshold_high)
        risk_level = RiskLevel(risk_str)
        cause = (
            "Motion pattern anomaly detected: spatiotemporal reconstruction error exceeded threshold."
            if risk_level != RiskLevel.NONE
            else "Normal scene motion."
        )
        samples.append(
            {
                "riskLevel": risk_level.value,
                "timeSeconds": t_sec,
                "loss": float(loss),
                "cause": cause,
            }
        )

        if first_alert_bunch_idx is None and risk_level in {RiskLevel.MEDIUM, RiskLevel.HIGH}:
            first_alert_bunch_idx = bunch_idx

        if stop_on_high and risk_level == RiskLevel.HIGH:
            break

    max_loss = float(np.max(losses)) if losses else 0.0
    mean_loss = float(np.mean(losses)) if losses else 0.0

    seconds_per_bunch = 10.0 * float(sample_every_seconds)
    event_time_seconds = 0.0
    if first_alert_bunch_idx is not None:
        event_time_seconds = float(first_alert_bunch_idx * seconds_per_bunch)

    risk_str = _classify_risk(max_loss, threshold_low, threshold_medium, threshold_high)
    risk_level = RiskLevel(risk_str)

    return AnalysisResult(
        risk_level=risk_level,
        risk_score=max_loss,
        max_loss=max_loss,
        mean_loss=mean_loss,
        event_time_seconds=event_time_seconds,
        samples=samples,
        losses=losses if include_losses else None,
    )
