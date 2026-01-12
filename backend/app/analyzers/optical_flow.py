from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from ..models import RiskLevel

@dataclass
class RiskSample:
    time_seconds: float
    risk_level: RiskLevel
    mean_flow_mag: float
    z_score: float
    active_ratio: float
    cause: str

@dataclass
class OpticalFlowAnalysisResult:
    risk_level: RiskLevel
    risk_score: float
    event_time_seconds: float
    samples: List[RiskSample]
    counts: dict

def _rolling_median_mad(values: list[float], window: int) -> tuple[float, float]:
    if len(values) == 0:
        return 0.0, 0.0
    w = values[-window:] if len(values) > window else values
    med = float(np.median(w))
    mad = float(np.median(np.abs(np.array(w) - med)))
    return med, mad

def _risk_from_z(z: float, z_low: float, z_med: float, z_high: float) -> RiskLevel:
    if z > z_high:
        return RiskLevel.HIGH
    if z > z_med:
        return RiskLevel.MEDIUM
    if z > z_low:
        return RiskLevel.LOW
    return RiskLevel.NONE

def _cause_for(risk: RiskLevel, *, z: float, active_ratio: float) -> str:
    if risk == RiskLevel.NONE:
        return "Normal scene motion."

    widespread = active_ratio >= 0.18

    if risk == RiskLevel.HIGH:
        if widespread:
            return "Sudden crowd acceleration combined with density spike (widespread scene-level motion)."
        return "Sudden crowd acceleration detected (optical-flow spike)."

    if risk == RiskLevel.MEDIUM:
        if widespread:
            return "Elevated crowd motion with widespread movement."
        return "Elevated crowd motion detected."

    if widespread:
        return "Noticeable motion increase across the scene."
    return "Noticeable motion spike detected."

def analyze_video_optical_flow(
    *,
    video_path: str,
    process_fps: float = 5.0,  
    resize_width: int = 320,
    mad_window: int = 30,
    z_low: float = 3.0,
    z_med: float = 5.0,
    z_high: float = 7.0,
    min_consecutive: int = 1,
    stop_on_high: bool = True,
    active_mag_threshold: float = 1.0,
) -> OpticalFlowAnalysisResult:
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0

    step = max(1, int(round(fps / max(process_fps, 0.1))))
    prev_gray = None
    mags: list[float] = []
    samples: List[RiskSample] = []
    counts = {"NONE": 0, "LOW": 0, "MEDIUM": 0, "HIGH": 0}
    overall_risk = RiskLevel.NONE
    first_high_time: Optional[float] = None
    consec = 0
    frame_idx = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if frame_idx % step != 0:
                frame_idx += 1
                continue
            t_sec = float(frame_idx / fps)

            h, w = frame_bgr.shape[:2]
            if w > 0 and resize_width > 0 and w != resize_width:
                new_w = int(resize_width)
                new_h = max(1, int(h * (new_w / w)))
                frame_bgr = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

            if prev_gray is None:
                prev_gray = gray
                frame_idx += 1
                continue

            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0,)
            mag, _ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mean_mag = float(np.mean(mag))
            active_ratio = float(np.mean(mag > float(active_mag_threshold)))
            mags.append(mean_mag)
            med, mad = _rolling_median_mad(mags[:-1], window=mad_window)
            denom = (mad * 1.4826) + 1e-6
            z = (mean_mag - med) / denom if len(mags) > 2 else 0.0

            risk = _risk_from_z(float(z), z_low, z_med, z_high)

            if risk != RiskLevel.NONE:
                consec += 1
            else:
                consec = 0

            escalated = risk
            if risk != RiskLevel.NONE and consec < max(1, int(min_consecutive)):
                escalated = RiskLevel.NONE

            cause = _cause_for(escalated, z=float(z), active_ratio=active_ratio)

            samples.append(RiskSample(time_seconds=t_sec, risk_level=escalated, mean_flow_mag=mean_mag, z_score=float(z), active_ratio=active_ratio, cause=cause,))
            counts[escalated.value] = counts.get(escalated.value, 0) + 1
            overall_risk = max(overall_risk, escalated, key=lambda r: ["NONE", "LOW", "MEDIUM", "HIGH"].index(r.value),)

            if escalated == RiskLevel.HIGH and first_high_time is None:
                first_high_time = t_sec
                if stop_on_high:
                    break

            prev_gray = gray
            frame_idx += 1

    finally:
        cap.release()

    event_time_seconds = float(first_high_time or 0.0)
    risk_score = float(first_high_time or 0.0)  

    return OpticalFlowAnalysisResult(risk_level=overall_risk, risk_score=risk_score, event_time_seconds=event_time_seconds, samples=samples, counts=counts,)
