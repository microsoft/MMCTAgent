"""Action boundary detection for keyframes using visual analysis."""

import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None


@dataclass
class BoundaryThresholds:
    """Configurable thresholds for boundary detection."""

    histogram_threshold: float = 0.4  # Chi-square distance threshold
    edge_threshold: float = 0.35  # Edge density change threshold
    motion_threshold: float = 2.0  # Motion magnitude threshold
    color_threshold: float = 0.3  # Color distribution change threshold


def detect_action_boundaries(
    keyframes: List[Dict[str, Any]],
    thresholds: Optional[BoundaryThresholds] = None,
) -> List[Dict[str, Any]]:
    """
    Detect action boundaries from keyframe sequence using multi-signal analysis.
    
    Boundaries are detected when:
    - Significant histogram/color change between frames (scene change)
    - Edge density changes significantly (composition change)
    - High motion between consecutive frames (action boundary)
    
    Args:
        keyframes: List of keyframe data with 'image' (numpy array) and 'timestamp'
        thresholds: Optional custom thresholds for detection
        
    Returns:
        List of boundary markers with timestamps, types, and confidence scores
    """
    if thresholds is None:
        thresholds = BoundaryThresholds()

    if cv2 is None:
        raise ImportError(
            "opencv-python (or opencv-python-headless) is required for boundary detection. "
            "Install with the `video-agent` extra."
        )

    boundaries = []

    if len(keyframes) < 2:
        return boundaries

    # Pre-compute features for all frames
    frame_features = []
    for kf in keyframes:
        img = kf.get("image")
        if img is not None:
            features = _extract_frame_features(img)
        else:
            features = None
        frame_features.append(features)

    for i in range(1, len(keyframes)):
        prev_features = frame_features[i - 1]
        curr_features = frame_features[i]

        if prev_features is None or curr_features is None:
            continue

        # Calculate multiple visual difference signals
        signals = _calculate_visual_signals(prev_features, curr_features)

        # Determine if this is a boundary and classify it
        boundary_info = _detect_and_classify_boundary(signals, thresholds)

        if boundary_info is not None:
            boundaries.append({
                "frame_index": i,
                "timestamp": keyframes[i].get("timestamp", 0),
                "boundary_type": boundary_info["type"],
                "confidence": boundary_info["confidence"],
                "signals": {
                    "histogram_diff": signals["histogram_diff"],
                    "edge_diff": signals["edge_diff"],
                    "motion_score": signals["motion_score"],
                    "color_diff": signals["color_diff"],
                },
            })

    return boundaries


def _extract_frame_features(image: np.ndarray) -> Dict[str, Any]:
    """
    Extract multiple features from a frame for boundary detection.
    
    Args:
        image: BGR numpy array
        
    Returns:
        Dictionary of extracted features
    """
    # Resize for consistent feature extraction
    target_size = (320, 240)
    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    # Convert to different color spaces
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

    # 1. Color histogram (HSV space - more robust to lighting)
    h_hist = cv2.calcHist([hsv], [0], None, [50], [0, 180])
    s_hist = cv2.calcHist([hsv], [1], None, [60], [0, 256])
    v_hist = cv2.calcHist([hsv], [2], None, [60], [0, 256])

    # Normalize histograms
    cv2.normalize(h_hist, h_hist)
    cv2.normalize(s_hist, s_hist)
    cv2.normalize(v_hist, v_hist)

    # 2. Edge features using Canny
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size

    # 3. Edge direction histogram (for composition analysis)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    direction = np.arctan2(sobely, sobelx)

    # Quantize directions into 8 bins
    dir_hist, _ = np.histogram(direction[magnitude > 20], bins=8, range=(-np.pi, np.pi))
    dir_hist = dir_hist.astype(np.float32)
    if dir_hist.sum() > 0:
        dir_hist /= dir_hist.sum()

    # 4. Dominant colors (k-means simplified to mean color per region)
    h, w = resized.shape[:2]
    regions = [
        resized[0 : h // 2, 0 : w // 2],
        resized[0 : h // 2, w // 2 :],
        resized[h // 2 :, 0 : w // 2],
        resized[h // 2 :, w // 2 :],
    ]
    region_colors = [np.mean(r, axis=(0, 1)) for r in regions]

    return {
        "h_hist": h_hist.flatten(),
        "s_hist": s_hist.flatten(),
        "v_hist": v_hist.flatten(),
        "edge_density": edge_density,
        "edge_dir_hist": dir_hist,
        "region_colors": np.array(region_colors),
        "gray": gray,  # Keep for motion estimation
    }


def _calculate_visual_signals(
    prev_features: Dict[str, Any],
    curr_features: Dict[str, Any],
) -> Dict[str, float]:
    """
    Calculate multiple visual difference signals between two frames.
    
    Returns:
        Dictionary of signal values
    """
    # 1. Histogram difference (chi-square distance for H, S, V)
    h_diff = cv2.compareHist(
        prev_features["h_hist"].reshape(-1, 1),
        curr_features["h_hist"].reshape(-1, 1),
        cv2.HISTCMP_CHISQR,
    )
    s_diff = cv2.compareHist(
        prev_features["s_hist"].reshape(-1, 1),
        curr_features["s_hist"].reshape(-1, 1),
        cv2.HISTCMP_CHISQR,
    )
    v_diff = cv2.compareHist(
        prev_features["v_hist"].reshape(-1, 1),
        curr_features["v_hist"].reshape(-1, 1),
        cv2.HISTCMP_CHISQR,
    )
    # Weighted combination (hue most important for scene detection)
    histogram_diff = 0.5 * h_diff + 0.3 * s_diff + 0.2 * v_diff

    # 2. Edge density difference
    edge_diff = abs(curr_features["edge_density"] - prev_features["edge_density"])

    # 3. Motion estimation using optical flow
    motion_score = _estimate_motion(prev_features["gray"], curr_features["gray"])

    # 4. Color distribution change
    color_diff = np.mean(
        np.abs(curr_features["region_colors"] - prev_features["region_colors"])
    ) / 255.0

    # 5. Edge direction change (composition shift)
    dir_diff = np.sum(
        np.abs(curr_features["edge_dir_hist"] - prev_features["edge_dir_hist"])
    ) / 2.0

    return {
        "histogram_diff": float(histogram_diff),
        "edge_diff": float(edge_diff),
        "motion_score": float(motion_score),
        "color_diff": float(color_diff),
        "dir_diff": float(dir_diff),
    }


def _estimate_motion(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    """
    Estimate motion magnitude between two grayscale frames using optical flow.
    
    Args:
        prev_gray: Previous frame (grayscale)
        curr_gray: Current frame (grayscale)
        
    Returns:
        Mean motion magnitude
    """
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return float(np.mean(magnitude))


def _detect_and_classify_boundary(
    signals: Dict[str, float],
    thresholds: BoundaryThresholds,
) -> Optional[Dict[str, Any]]:
    """
    Determine if signals indicate a boundary and classify its type.
    
    Args:
        signals: Dictionary of visual signals
        thresholds: Detection thresholds
        
    Returns:
        Boundary info dict or None if no boundary detected
    """
    hist_triggered = signals["histogram_diff"] > thresholds.histogram_threshold
    edge_triggered = signals["edge_diff"] > thresholds.edge_threshold
    motion_triggered = signals["motion_score"] > thresholds.motion_threshold
    color_triggered = signals["color_diff"] > thresholds.color_threshold

    # No boundary if none triggered
    if not any([hist_triggered, edge_triggered, motion_triggered, color_triggered]):
        return None

    # Classify boundary type based on which signals triggered
    # Priority: scene_change > action_start/end > composition_change
    if hist_triggered and color_triggered:
        boundary_type = "scene_change"
        confidence = (signals["histogram_diff"] + signals["color_diff"]) / 2.0
        confidence = min(1.0, confidence)
    elif motion_triggered:
        # High motion with edge change suggests action boundary
        if edge_triggered:
            boundary_type = "action_boundary"
        else:
            boundary_type = "motion_peak"
        confidence = min(1.0, signals["motion_score"] / (thresholds.motion_threshold * 2))
    elif edge_triggered:
        boundary_type = "composition_change"
        confidence = min(1.0, signals["edge_diff"] / thresholds.edge_threshold)
    elif hist_triggered:
        boundary_type = "lighting_change"
        confidence = min(1.0, signals["histogram_diff"] / thresholds.histogram_threshold)
    else:
        boundary_type = "subtle_change"
        confidence = 0.5

    return {
        "type": boundary_type,
        "confidence": float(confidence),
    }
