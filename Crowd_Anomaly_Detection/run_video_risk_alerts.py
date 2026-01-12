import argparse
import os
from datetime import timedelta
import numpy as np

def _format_hhmmss(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    return str(timedelta(seconds=int(seconds))).rjust(8, "0")


def _mean_euclidean_loss(x1: np.ndarray, x2: np.ndarray) -> float:
    diff = x1 - x2
    n_samples = diff.size
    dist = np.sqrt(np.square(diff).sum())
    return float(dist / n_samples)


def _load_model(model_path: str):
    from tensorflow.keras.layers import Conv3D, Conv3DTranspose, ConvLSTM2D
    from tensorflow.keras.models import Sequential, load_model

    try:
        return load_model(model_path, compile=False)
    except Exception:
        model = Sequential()
        model.add(Conv3D(filters=128,  kernel_size=(11, 11, 1),  strides=(4, 4, 1),  padding="valid",  input_shape=(227, 227, 10, 1),  activation="tanh",))
        model.add(Conv3D(filters=64, kernel_size=(5, 5, 1), strides=(2, 2, 1), padding="valid", activation="tanh",))
        model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), strides=1, padding="same", dropout=0.4, recurrent_dropout=0.3, return_sequences=True,))
        model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), strides=1, padding="same", dropout=0.3, return_sequences=True,))
        model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), strides=1, padding="same", dropout=0.5, return_sequences=True,))
        model.add(Conv3DTranspose(filters=128, kernel_size=(5, 5, 1), strides=(2, 2, 1), padding="valid", activation="tanh",))
        model.add(Conv3DTranspose(filters=1, kernel_size=(11, 11, 1), strides=(4, 4, 1), padding="valid", activation="tanh",))
        model.load_weights(model_path)
        return model
    
def _extract_sampled_grayscale_frames(video_path: str, sample_every_seconds: float) -> list[np.ndarray]:
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0

    step_frames = max(1, int(round(sample_every_seconds * fps)))

    frames_gray: list[np.ndarray] = []
    frame_idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        if frame_idx % step_frames == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (227, 227), interpolation=cv2.INTER_AREA)
            gray = (0.2989 * frame_rgb[:, :, 0] + 0.5870 * frame_rgb[:, :, 1] + 0.1140 * frame_rgb[:, :, 2]).astype(np.float32)
            frames_gray.append(gray)

        frame_idx += 1

    cap.release()
    return frames_gray


def _preprocess_to_model_tensor(frames_gray: list[np.ndarray]) -> tuple[np.ndarray, int]:
    if len(frames_gray) < 10:
        raise RuntimeError(f"Need at least 10 sampled frames; got {len(frames_gray)}")

    arr = np.stack(frames_gray, axis=0).transpose(1, 2, 0)

    mean = float(arr.mean())
    std = float(arr.std())
    if std == 0:
        std = 1.0

    arr = (arr - mean) / std
    arr = np.clip(arr, 0, 1)

    total_frames = int(arr.shape[2])
    usable_frames = total_frames - (total_frames % 10)
    if usable_frames < 10:
        raise RuntimeError("Not enough frames after trimming to multiples of 10")

    arr = arr[:, :, :usable_frames]
    bunches = arr.reshape(227, 227, -1, 10).transpose(2, 0, 1, 3)
    bunches = np.expand_dims(bunches, axis=4)

    return bunches.astype(np.float32), usable_frames


def _classify_risk(loss: float, t_low: float, t_med: float, t_high: float) -> str:
    if loss > t_high:
        return "HIGH"
    if loss > t_med:
        return "MEDIUM"
    if loss > t_low:
        return "LOW"
    return "NONE"


def main() -> int:
    parser = argparse.ArgumentParser(description=("Run the pretrained crowd anomaly model on a video and print risk alerts " "(crowd-level only; no face recognition / no identity tracking)."))
    parser.add_argument("--video", required=True, help="Path to input video file (mp4/avi/etc).")
    parser.add_argument("--model", default=None, help="Path to .h5 model. Defaults to '(Pretrained)AnomalyDetector.h5' if present.",)
    parser.add_argument("--sample-every-seconds", type=float, default=5.0, help="Sample 1 frame every N seconds (default: 5.0).",)
    parser.add_argument("--print-scores", action="store_true", help="Print the raw reconstruction loss for every 10-frame bunch.",)
    parser.add_argument("--auto-thresholds", action="store_true", help=("Auto-calibrate thresholds from this video's loss distribution using percentiles. " "Useful when losses are consistently below MEDIUM/HIGH fixed thresholds."),)
    parser.add_argument("--auto-low-percentile", type=float, default=90.0, help="Percentile used for LOW threshold when --auto-thresholds is set (default: 90).",)
    parser.add_argument("--auto-medium-percentile", type=float, default=97.0, help="Percentile used for MEDIUM threshold when --auto-thresholds is set (default: 97).",)
    parser.add_argument("--auto-high-percentile", type=float, default=99.0, help="Percentile used for HIGH threshold when --auto-thresholds is set (default: 99).",)
    parser.add_argument("--threshold-low", type=float, default=0.0008, help="LOW threshold.")
    parser.add_argument("--threshold-medium", type=float, default=0.0012, help="MEDIUM threshold (must be >= low).",)
    parser.add_argument("--threshold-high", type=float, default=0.0016, help="HIGH threshold (must be >= medium).",)
    args = parser.parse_args()

    video_path = os.path.abspath(args.video)
    if not os.path.exists(video_path):
        raise SystemExit(f"Video not found: {video_path}")

    repo_dir = os.path.dirname(os.path.abspath(__file__))
    default_model = os.path.join(repo_dir, "(Pretrained)AnomalyDetector.h5")
    model_path = os.path.abspath(args.model) if args.model else default_model
    if not os.path.exists(model_path):
        raise SystemExit("Model file not found. Expected '(Pretrained)AnomalyDetector.h5' in the repo, " "or pass --model path\\to\\AnomalyDetector.h5")

    if args.auto_thresholds:
        for name, pct in (
            ("auto-low-percentile", args.auto_low_percentile),
            ("auto-medium-percentile", args.auto_medium_percentile),
            ("auto-high-percentile", args.auto_high_percentile),
        ):
            if pct <= 0 or pct >= 100:
                raise SystemExit(f"{name} must be between 0 and 100")
        if not (args.auto_low_percentile <= args.auto_medium_percentile <= args.auto_high_percentile):
            raise SystemExit("Auto percentiles must satisfy low <= medium <= high")
    else:
        if not (args.threshold_low <= args.threshold_medium <= args.threshold_high):
            raise SystemExit("Thresholds must satisfy low <= medium <= high")

    frames_gray = _extract_sampled_grayscale_frames(video_path, args.sample_every_seconds)
    bunches, usable_frames = _preprocess_to_model_tensor(frames_gray)
    model = _load_model(model_path)
    seconds_per_bunch = 10.0 * args.sample_every_seconds
    losses: list[float] = []
    for bunch in bunches:
        n_bunch = np.expand_dims(bunch, axis=0)  
        reconstructed = model.predict(n_bunch, verbose=0)
        losses.append(_mean_euclidean_loss(n_bunch, reconstructed))

    if args.auto_thresholds:
        t_low = float(np.quantile(losses, args.auto_low_percentile / 100.0))
        t_med = float(np.quantile(losses, args.auto_medium_percentile / 100.0))
        t_high = float(np.quantile(losses, args.auto_high_percentile / 100.0))
        t_med = max(t_med, t_low)
        t_high = max(t_high, t_med)
        print("Auto thresholds " f"(p{args.auto_low_percentile:g}/p{args.auto_medium_percentile:g}/p{args.auto_high_percentile:g}): " f"LOW={t_low:.6f}, MEDIUM={t_med:.6f}, HIGH={t_high:.6f}")
    else:
        t_low, t_med, t_high = args.threshold_low, args.threshold_medium, args.threshold_high
    
    any_alert = False
    counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "NONE": 0}
    for bunch_idx, loss in enumerate(losses):
        timestamp = _format_hhmmss(bunch_idx * seconds_per_bunch)
        if args.print_scores:
            print(f"Score: {loss:.6f}  Time: {timestamp}")

        risk = _classify_risk(loss, t_low, t_med, t_high)
        counts[risk] = counts.get(risk, 0) + 1

        if risk != "NONE":
            any_alert = True
            explanation = ("Motion pattern anomaly detected: spatiotemporal reconstruction error " f"{loss:.6f} exceeded threshold for {risk}.")
            print(f"Time: {timestamp}")
            print(f"Risk: {risk}")
            print(f"Explanation: {explanation}")
            print("-")

    print("Summary")
    print(f"- Sampled frames: {len(frames_gray)} (usable: {usable_frames})")
    print(f"- Bunches analyzed: {len(bunches)}")
    print(f"- Alerts: LOW={counts['LOW']}, MEDIUM={counts['MEDIUM']}, HIGH={counts['HIGH']}")
    if not any_alert:
        print("- No anomalies detected (at configured thresholds).")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
