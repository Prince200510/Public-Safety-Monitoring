import argparse
import os
from datetime import timedelta
import numpy as np

def _format_hhmmss(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    return str(timedelta(seconds=int(seconds))).rjust(8, "0")

def _rolling_median_mad(values: list[float], window: int) -> tuple[float, float]:
    if len(values) == 0:
        return 0.0, 0.0
    w = values[-window:] if len(values) > window else values
    med = float(np.median(w))
    mad = float(np.median(np.abs(np.array(w) - med)))
    return med, mad

def _risk_from_z(z: float, z_low: float, z_med: float, z_high: float) -> str:
    if z > z_high:
        return "HIGH"
    if z > z_med:
        return "MEDIUM"
    if z > z_low:
        return "LOW"
    return "NONE"

def main() -> int:
    parser = argparse.ArgumentParser(description=("Optical-flow spike based risk alerts (crowd-level / scene motion only; " "no person ID, no face recognition)."))
    parser.add_argument("--video", required=True, help="Path to input video file.")
    parser.add_argument("--process-fps", type=float, default=10.0, help="How many frames-per-second to process for flow (default: 10).",)
    parser.add_argument("--resize-width", type=int, default=320, help="Resize width for flow computation (default: 320).",)
    parser.add_argument("--mad-window", type=int, default=30, help="Rolling window size for median/MAD baseline (default: 30 points).",)
    parser.add_argument("--min-consecutive", type=int, default=2, help="Require N consecutive triggered points before alerting (default: 2).",)
    parser.add_argument("--z-low", type=float, default=3.0, help="LOW z-score threshold.")
    parser.add_argument("--z-med", type=float, default=5.0, help="MEDIUM z-score threshold.")
    parser.add_argument("--z-high", type=float, default=7.0, help="HIGH z-score threshold.")
    parser.add_argument("--print-scores", action="store_true", help="Print per-sample flow magnitude and z-score.",)
    args = parser.parse_args()

    if not (args.z_low <= args.z_med <= args.z_high):
        raise SystemExit("z thresholds must satisfy low <= med <= high")

    video_path = os.path.abspath(args.video)
    if not os.path.exists(video_path):
        raise SystemExit(f"Video not found: {video_path}")

    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0

    step = max(1, int(round(fps / max(args.process_fps, 0.1))))

    prev_gray = None
    prev_t = None

    mags: list[float] = []
    consec = 0
    any_alert = False
    counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}

    frame_idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        if frame_idx % step != 0:
            frame_idx += 1
            continue

        t_sec = float(frame_idx / fps)
        h, w = frame_bgr.shape[:2]
        if w > 0 and args.resize_width > 0 and w != args.resize_width:
            new_w = int(args.resize_width)
            new_h = max(1, int(h * (new_w / w)))
            frame_bgr = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if prev_gray is None:
            prev_gray = gray
            prev_t = t_sec
            frame_idx += 1
            continue

        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0,)
        mag, _ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mean_mag = float(np.mean(mag))

        mags.append(mean_mag)
        med, mad = _rolling_median_mad(mags[:-1], window=args.mad_window)
        denom = (mad * 1.4826) + 1e-6
        z = (mean_mag - med) / denom if len(mags) > 2 else 0.0

        risk = _risk_from_z(z, args.z_low, args.z_med, args.z_high)

        if args.print_scores:
            print(f"FlowMag: {mean_mag:.4f}  z: {z:.2f}  Time: {_format_hhmmss(t_sec)}")

        if risk != "NONE":
            consec += 1
        else:
            consec = 0

        if risk != "NONE" and consec >= max(1, args.min_consecutive):
            any_alert = True
            counts[risk] = counts.get(risk, 0) + 1
            timestamp = _format_hhmmss(t_sec)
            explanation = (
                f"Optical-flow spike detected (z={z:.2f}). "
                "This indicates sudden acceleration/chaotic motion at scene level."
            )
            print(f"Time: {timestamp}")
            print(f"Risk: {risk}")
            print(f"Explanation: {explanation}")
            print("-")
            consec = 0  

        prev_gray = gray
        prev_t = t_sec
        frame_idx += 1

    cap.release()

    print("Summary")
    print(f"- Process FPS: {args.process_fps} (step={step} at source fps~{fps:.2f})")
    print(f"- Samples: {len(mags)}")
    print(f"- Alerts: LOW={counts.get('LOW',0)}, MEDIUM={counts.get('MEDIUM',0)}, HIGH={counts.get('HIGH',0)}")
    if not any_alert:
        print("- No motion anomalies detected (at configured thresholds).")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
