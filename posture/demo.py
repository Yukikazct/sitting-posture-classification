"""Single-person webcam/video demo; all heavyweight dependencies load on demand."""

from pathlib import Path
import time
from typing import Any, Dict, Mapping, Optional, Union

import numpy as np

from .temporal import ReminderState


CLASS_NAMES = ("upright", "head_down", "forward_lean", "side_lean")
SKELETON_EDGES = (
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
)


def _load_classifier(checkpoint_path, requested_device):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required. Install the project with: pip install -e .") from exc
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError("Train a classifier first; checkpoint not found: {}".format(checkpoint_path))
    from .evaluation import load_classifier
    model, mean, std, _, device = load_classifier(str(checkpoint_path), requested_device)
    return torch, model, mean, std, device


def _draw_skeleton(cv2, frame, keypoints, threshold):
    for first, second in SKELETON_EDGES:
        if min(keypoints[first, 2], keypoints[second, 2]) >= threshold:
            start = tuple(int(round(value)) for value in keypoints[first, :2])
            end = tuple(int(round(value)) for value in keypoints[second, :2])
            cv2.line(frame, start, end, (90, 210, 255), 2, cv2.LINE_AA)
    for x, y, confidence in keypoints:
        if confidence >= threshold:
            cv2.circle(frame, (int(round(x)), int(round(y))), 3, (70, 240, 100), -1)


def run_demo(
    checkpoint_path: Union[str, Path],
    config: Mapping[str, Any],
    source: Union[str, int] = "0",
    device: str = "auto",
    save_path: Optional[Union[str, Path]] = None,
    smoothed: bool = True,
) -> Dict[str, Any]:
    """Run inference without training or implicitly downloading pose weights.

    A numeric source opens that camera unless it names an existing file. Video
    time is frame_index / source_fps; webcam time is monotonic elapsed seconds.
    Exactly one visible pose is required; missing/occluded/multiple people reset
    reminders to unknown. ``q`` exits and ``r`` toggles reminders. Set
    ``config['demo']['display'] = False`` for a video run without GUI windows.
    """
    pose_settings = config.get("pose", {})
    weights_path = Path(pose_settings.get("weights", "models/yolo11n-pose.pt"))
    if not weights_path.is_file():
        raise FileNotFoundError(
            "Pose weights not found: {}. Obtain yolo11n-pose.pt explicitly and set pose.weights; "
            "the demo does not download weights.".format(weights_path)
        )
    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError("Train a classifier first; checkpoint not found: {}".format(checkpoint_path))
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Demo dependencies are missing. Install them with: pip install -e '.[vision]'"
        ) from exc
    from .features import extract_features

    source_text = str(source)
    camera = isinstance(source, int) or (source_text.isdecimal() and not Path(source_text).is_file())
    capture_source = int(source) if camera else source_text
    if not camera and not Path(source_text).is_file():
        raise FileNotFoundError("Input video not found: {}".format(source_text))
    output_path = Path(save_path) if save_path is not None else None
    if output_path is not None:
        if not camera and output_path.resolve() == Path(source_text).resolve():
            raise ValueError("Output video must differ from the input video.")
        if output_path.exists():
            raise FileExistsError("Choose a new output filename: {}".format(output_path))

    torch, classifier, mean, std, selected_device = _load_classifier(checkpoint_path, device)
    pose_model = YOLO(str(weights_path))
    pose_model.model.requires_grad_(False)
    pose_model.model.eval()
    state = ReminderState(config, smoothed=smoothed)
    threshold = float(pose_settings.get("min_keypoint_confidence", 0.3))
    display = bool(config.get("demo", {}).get("display", True))
    reminders_enabled = bool(config.get("demo", {}).get("reminders_enabled", True))
    capture = cv2.VideoCapture(capture_source)
    writer = None
    frames = 0
    reminder_events = 0
    unknown_frames = 0
    start_time = time.monotonic()
    try:
        if not capture.isOpened():
            raise RuntimeError("Cannot open camera/video: {}".format(source_text))
        input_fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not camera and (not np.isfinite(input_fps) or input_fps <= 0):
            raise ValueError("Input video has no reliable FPS; convert it to a constant-frame-rate video first.")
        output_fps = input_fps if np.isfinite(input_fps) and input_fps > 0 else 30.0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = time.monotonic() - start_time if camera else frames / input_fps
            probabilities = None
            reason = "No reliable person"
            results = pose_model.predict(
                source=frame, imgsz=int(pose_settings.get("imgsz", 640)),
                conf=float(pose_settings.get("person_confidence", 0.25)),
                device=str(selected_device), verbose=False,
            )
            if results and results[0].keypoints is not None:
                detections = results[0].keypoints.data.detach().cpu().numpy()
                if detections.shape == (1, 17, 3):
                    keypoints = detections[0]
                    features = extract_features(keypoints, min_confidence=threshold)
                    if features is not None:
                        inputs = torch.from_numpy(((features - mean) / std).astype(np.float32)[None, :]).to(selected_device)
                        with torch.inference_mode():
                            probabilities = torch.softmax(classifier(inputs), dim=1)[0].cpu().numpy()
                        _draw_skeleton(cv2, frame, keypoints, threshold)
                        reason = "Low class confidence"
                    else:
                        reason = "Required keypoints occluded"
                elif len(detections) > 1:
                    reason = "Multiple people: use one person"
            result = state.update(probabilities, timestamp)
            frames += 1
            unknown_frames += int(result["label"] < 0)
            reminder_events += int(result["alert"] and reminders_enabled)
            label = CLASS_NAMES[result["label"]] if result["label"] >= 0 else "unknown"
            color = (70, 80, 245) if result["active"] and reminders_enabled else (80, 220, 100)
            cv2.putText(frame, "{}  {:.2f}".format(label, result["confidence"]), (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            text = "Reminder {} | {} {:.1f}s | q: quit, r: toggle".format(
                "ON" if reminders_enabled else "OFF", result["status"], result["duration"])
            cv2.putText(frame, text, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            if result["label"] < 0:
                cv2.putText(frame, reason, (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 180, 255), 1)
            elif result["active"] and reminders_enabled:
                cv2.putText(frame, "Please adjust your posture", (15, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            if output_path is not None:
                if writer is None:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    codec = "MJPG" if output_path.suffix.lower() == ".avi" else "mp4v"
                    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*codec),
                                             output_fps, (frame.shape[1], frame.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError("Cannot create output video: {}".format(output_path))
                writer.write(frame)
            if display:
                cv2.imshow("Posture classification", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    reminders_enabled = not reminders_enabled
                    state.reset()
        if frames == 0:
            raise RuntimeError("Input opened but no frames could be read.")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()

    elapsed = max(time.monotonic() - start_time, 1e-9)
    return {
        "source": source_text, "device": str(selected_device), "frames": frames,
        "elapsed_seconds": elapsed, "fps": frames / elapsed,
        "reminder_events": reminder_events, "unknown_frames": unknown_frames,
        "unknown_fraction": unknown_frames / frames,
        "save_path": str(output_path) if output_path is not None else None,
    }
