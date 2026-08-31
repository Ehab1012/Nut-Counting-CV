"""Nut detection and counting with Gaussian -> grayscale -> Canny -> watershed detection,
plus Kalman-filtered multi-frame tracking and offline trajectory smoothing for a stable,
non-jittery rendered output.

The per-frame detector is unchanged from the original: Canny edges find nut boundaries,
a color gate rejects blue conveyor pixels, and watershed markers separate touching nuts
(with a Hough-circle refinement pass on top). That part was already carefully tuned, so
it is left alone. The jitter lived in what happened *after* detection -- boxes flickering
on/off and shaking frame to frame -- so that is where this version does the real work:

  1) Pass 1 (detect):  run the tuned detector on every frame, unchanged.
  2) Pass 2 (track):   associate detections into tracks using a Kalman filter (constant-
     velocity model over cx, cy, w, h) instead of a raw two-frame velocity estimate. Every
     active track is predicted every frame, even ones the detector misses that frame, so a
     track glides through brief dropouts instead of freezing or disappearing.
  3) Pass 3 (confirm):  a track must accumulate MIN_HITS real detections before it is
     trusted as a nut rather than a one-off noise blob. Because this is offline, once a
     track is confirmed its *entire* life is rendered -- including the frames before it
     reached MIN_HITS -- so trustworthy tracks don't "pop in" late. Only tracks that never
     reach MIN_HITS (i.e. look like noise) are dropped, and they're dropped completely
     rather than flickering on screen for a frame or two.
  4) Pass 4 (smooth):  each confirmed track's per-frame (cx, cy, w, h) series is smoothed
     with a zero-phase (non-causal) Gaussian-weighted moving average. Because the whole
     video is available up front, this can use both past AND future samples of the track,
     which removes residual shake that even the causal Kalman estimate leaves in.
  5) Pass 5 (count):   the line crossing is detected on the *smoothed* trajectory, which
     removes false/duplicate crossing triggers caused by raw positional noise near the
     line. The exact crossing frame is recorded so the on-screen counter still increments
     at a precise moment rather than lagging.
  6) Pass 6 (render):  the source video is re-read and annotated using the precomputed
     smoothed boxes/centroids and crossing events.

Only classical CV techniques are used throughout (Canny, watershed, Hough circles, Kalman
filtering, Gaussian convolution) -- no learned/deep models, matching the original design.

Revision note (after reviewing output on real footage): the first version of this
pipeline put box width/height into the Kalman state alongside a "velocity" for each.
On real footage, a single bad detection match (e.g. a stray blob picked up near a hand)
implied a large one-frame size change; the filter treated that as a genuine rate of
change and kept extrapolating it every subsequent coasted frame, so a ~20px box could
balloon to hundreds of pixels within the max-missed window -- the "flying box" bug. Box
size is now tracked with a bounded exponential moving average instead of a Kalman
velocity term (see KalmanPointTracker's docstring), and detection-to-track matching now
also gates on size plausibility (SIZE_RATIO_MIN/MAX), so a track can no longer latch onto
a wrong-sized region at all. A hard render-time size ceiling (MAX_RENDER_BOX_SIDE) is
kept as a last-resort backstop.
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

# Detection: (x, y, width, height, (centroid_x, centroid_y))
# Centroid is kept as sub-pixel float (not rounded to int) so it doesn't inject its own
# +/-0.5px quantization jitter into the tracker/smoother downstream.
Detection = Tuple[int, int, int, int, Tuple[float, float]]

# Pre-allocated OpenCV structuring elements for morphology operations
KERNEL_ELLIPSE_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
KERNEL_ELLIPSE_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
KERNEL_RECT_3 = np.ones((3, 3), np.uint8)

# Precomputed trigonometric unit offsets for 8 circular direction samples
ANGLES_RAD = np.deg2rad(np.arange(0, 360, 45))
COS_ANGLES = np.cos(ANGLES_RAD)
SIN_ANGLES = np.sin(ANGLES_RAD)


def edge_watershed_detections(
    frame: np.ndarray, min_area: int = 55, max_area: int = 950, min_separation: int = 9
) -> List[Detection]:
    """Extract individual nuts using Gaussian blur, grayscale, Canny, and watershed.

    Candidate-mask construction (Canny + color gates, up through `candidate` below) is
    unchanged from the original tuning -- see module docstring for why. The watershed
    marker/flooding stage below it has been reworked for better separation of touching
    or clustered nuts -- see the comments in that section for what changed and why.
    """
    # 1) Gaussian smoothing, 2) grayscale, 3) Canny edges.
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 45, 125)

    # Convert B channel once to float32 for ratio calculations
    b, g, r = cv2.split(blurred)
    b_float = b.astype(np.float32)

    # Reject blue conveyor background gate (relaxed thresholds for side-profile & shadowed nuts)
    non_blue = (
        (r > 70)
        & (g > 50)
        & (r > b_float * 0.40)
        & (g > b_float * 0.36)
    ).astype(np.uint8) * 255

    edge_fg = cv2.bitwise_and(edges, non_blue)
    edge_fg = cv2.dilate(edge_fg, KERNEL_ELLIPSE_3, iterations=1)
    edge_fg = cv2.morphologyEx(edge_fg, cv2.MORPH_CLOSE, KERNEL_ELLIPSE_5, iterations=2)

    # Create filled object candidates from Canny contours
    contours, _ = cv2.findContours(edge_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidate = np.zeros_like(edge_fg)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if 4 <= w <= 65 and 4 <= h <= 65:
            cv2.drawContours(candidate, [contour], -1, 255, thickness=cv2.FILLED)

    # Keep only pixels supported by pale nut appearance
    pale = (
        (r > 75)
        & (g > 52)
        & (r > b_float * 0.41)
        & (g > b_float * 0.37)
    ).astype(np.uint8) * 255
    candidate = cv2.bitwise_and(candidate, cv2.dilate(pale, KERNEL_RECT_3, iterations=1))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, KERNEL_ELLIPSE_5, iterations=1)

    # 4) Watershed markers, from regional maxima of the distance transform.
    #
    # The previous version thresholded the whole distance map by one relative cutoff
    # (`dist > 0.24 * max_dist`). For an isolated nut that's fine -- but when two nuts
    # touch, the "above cutoff" region of each can itself touch the other's, so
    # connectedComponents fuses them into a single marker and watershed never gets a
    # chance to split them (classic watershed under-segmentation of touching objects).
    #
    # The fix is the standard one for this exact problem (the same idea as
    # skimage.feature.peak_local_max): instead of keeping the whole thresholded region,
    # keep only each region's REGIONAL MAXIMUM -- the single peak (nut center) -- as its
    # marker seed. Two touching nuts still have two separate peaks even though their
    # thresholded blobs merge, so they now get two markers and watershed correctly draws
    # a boundary between them. `min_separation` is the minimum center-to-center distance
    # (px) at which two touching nuts are still resolved as separate peaks; below that,
    # they merge into one (an inherent limit of centroid/distance-based separation, not
    # something thresholding differently can fix). Real nuts in this footage measured
    # ~20px across, so two touching nuts sit ~18-22px apart center-to-center -- the
    # default of 9 comfortably resolves that while still ignoring single-pixel noise
    # bumps (a peak has to be the true local max within its own 9x9 neighborhood).
    dist = cv2.distanceTransform(candidate, cv2.DIST_L2, 5)
    max_dist = dist.max()
    if max_dist <= 0:
        return []

    # Light smoothing so compression/sensor noise doesn't fabricate extra tiny peaks.
    dist_smooth = cv2.GaussianBlur(dist, (3, 3), 0)

    # Regional-maxima extraction via the classical "dilate and compare" trick: a pixel
    # survives only if it equals the max of its own local_separation-sized neighborhood.
    sep = max(3, min_separation | 1)  # must be odd and at least 3
    peak_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sep, sep))
    local_max = cv2.dilate(dist_smooth, peak_kernel)
    peak_floor = max(1.8, 0.24 * max_dist)  # keep the original sensitivity tuning
    is_peak = (dist_smooth >= local_max) & (dist_smooth > peak_floor)
    sure_fg = is_peak.astype(np.uint8) * 255
    # Thicken each (possibly single-pixel) peak slightly so connectedComponents gives a
    # small, stable marker blob per nut rather than a fragile 1px seed.
    sure_fg = cv2.dilate(sure_fg, KERNEL_ELLIPSE_3, iterations=1)

    marker_count, markers = cv2.connectedComponents(sure_fg)

    # A small margin around the raw candidate mask (the "sure background") gives
    # watershed a little extra room to place the boundary between two touching nuts
    # rather than pinning it to the exact, possibly slightly eroded, candidate edge.
    sure_bg = cv2.dilate(candidate, KERNEL_ELLIPSE_3, iterations=1)
    unknown = cv2.subtract(sure_bg, sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Flood on an explicit gradient image rather than raw color. The seam between two
    # touching same-colored nuts is usually a faint highlight/shadow line; a
    # morphological gradient of the grayscale image brings that out more cleanly than
    # raw BGR, which is otherwise dominated by the strong (and here irrelevant) belt-vs-
    # nut color contrast.
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, KERNEL_ELLIPSE_3)
    gradient_bgr = cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)
    cv2.watershed(gradient_bgr, markers)

    h_img, w_img = frame.shape[:2]
    detections: List[Detection] = []

    for marker in range(2, marker_count + 1):
        region = np.uint8(markers == marker) * 255
        area = cv2.countNonZero(region)
        if not (min_area <= area <= max_area):
            continue

        region_contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not region_contours:
            continue

        contour = max(region_contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        if not (4 <= w <= 55 and 4 <= h <= 55):
            continue

        # Border filtering: ignore partial objects touching frame boundaries
        if x <= 2 or y <= 2 or x + w >= w_img - 2 or y + h >= h_img - 50:
            continue

        fill_ratio = area / float(w * h)
        aspect = min(w, h) / float(max(w, h))
        # Lower aspect ratio threshold (0.25 instead of 0.48) to capture vertical/elongated nuts
        if fill_ratio < 0.22 or aspect < 0.25:
            continue

        m = cv2.moments(contour)
        if m["m00"] == 0:
            continue

        # Sub-pixel centroid (not rounded) -- see Detection type comment above.
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        detections.append((x, y, w, h, (cx, cy)))

    # Suppress duplicate/nested detection boxes
    kept: List[Detection] = []
    for det in sorted(detections, key=lambda d: d[2] * d[3], reverse=True):
        x, y, w, h, (cx, cy) = det
        duplicate = False
        for ox, oy, ow, oh, (ocx, ocy) in kept:
            # Fast squared distance check (< 8 px -> dist^2 < 64)
            if (cx - ocx) ** 2 + (cy - ocy) ** 2 < 64:
                duplicate = True
                break
            iw = max(0, min(x + w, ox + ow) - max(x, ox))
            ih = max(0, min(y + h, oy + oh) - max(y, oy))
            inter = iw * ih
            union = w * h + ow * oh - inter
            if union > 0 and (inter / union) > 0.40:
                duplicate = True
                break
        if not duplicate:
            kept.append(det)

    # Hough circles refinement for ring-shaped nuts
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=13,
        param1=80,
        param2=13,
        minRadius=6,
        maxRadius=18,
    )

    proposals = list(kept)
    if circles is not None:
        for cx_f, cy_f, r_f in circles[0]:
            # Integer pixel coords only for indexing into the image; the sub-pixel
            # (cx_f, cy_f) from HoughCircles is kept for the stored detection.
            cx_i, cy_i, radius = int(round(cx_f)), int(round(cy_f)), int(round(r_f))
            if radius <= 0:
                continue

            # Vectorized sample locations around ring
            sx = np.clip((cx_i + COS_ANGLES * radius).astype(int), 0, w_img - 1)
            sy = np.clip((cy_i + SIN_ANGLES * radius).astype(int), 0, h_img - 1)

            # Extract sample pixel colors (8, 3) and take mean color (BGR)
            sample = blurred[sy, sx].mean(axis=0)
            if not (sample[2] > sample[0] * 0.48 and sample[1] > sample[0] * 0.42):
                continue

            circle_det = (
                max(0, cx_i - radius),
                max(0, cy_i - radius),
                2 * radius,
                2 * radius,
                (float(cx_f), float(cy_f)),
            )

            # Replace nearby watershed blob with circle proposal (dist < 12 px -> dist^2 < 144)
            nearby_idx = None
            for i, d in enumerate(proposals):
                if (d[4][0] - cx_f) ** 2 + (d[4][1] - cy_f) ** 2 < 144:
                    nearby_idx = i
                    break

            if nearby_idx is not None:
                proposals[nearby_idx] = circle_det
            else:
                proposals.append(circle_det)

    # Final non-maximum suppression (dist < 10 px -> dist^2 < 100)
    final: List[Detection] = []
    for det in sorted(proposals, key=lambda d: d[2] * d[3], reverse=True):
        cx, cy = det[4]
        if not any((cx - old[4][0]) ** 2 + (cy - old[4][1]) ** 2 < 100 for old in final):
            final.append(det)

    return final


class KalmanPointTracker:
    """Constant-velocity Kalman filter over a track's (center_x, center_y) position ONLY.

    Earlier versions of this file also put box width/height into the Kalman state with
    their own "velocity" terms. On real footage that caused exactly the "flying giant
    box" failure: a single bad association (e.g. a track briefly grabbing a same-sized-
    but-wrong blob) implies a large one-frame change in w/h, the filter reads that as a
    real rate of change, and then every subsequent *predicted* (coasted) frame keeps
    adding that same bad rate again -- a handful of coasted frames is enough to blow a
    ~20px box up to hundreds of pixels. Position genuinely has momentum (a nut moving on
    a belt), so a velocity model is appropriate there and is kept. Apparent box size does
    not have "momentum" the same way, so it is now tracked separately with a bounded
    exponential moving average (see Track.w / Track.h below) that can never run away,
    because it is always a weighted average of values the detector itself already caps.
    A hard velocity clamp (below) adds a second, independent guard on the position side.
    """

    MEASUREMENT_NOISE = 6.0
    POSITION_PROCESS_NOISE = 0.5
    VELOCITY_PROCESS_NOISE = 4.0
    # Real footage showed nuts moving ~3-5 px/frame typically. 25 px/frame is a generous
    # multiple of that -- comfortably covers genuinely fast motion while still bounding
    # how far a single bad association can make a track "fly" per frame.
    MAX_VELOCITY = 25.0

    def __init__(self, cx: float, cy: float):
        kf = cv2.KalmanFilter(4, 2)
        kf.transitionMatrix = np.array(
            [
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        kf.measurementMatrix = np.eye(2, 4, dtype=np.float32)
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * self.MEASUREMENT_NOISE
        process_noise = np.eye(4, dtype=np.float32)
        process_noise[:2, :2] *= self.POSITION_PROCESS_NOISE
        process_noise[2:, 2:] *= self.VELOCITY_PROCESS_NOISE
        kf.processNoiseCov = process_noise
        kf.errorCovPost = np.eye(4, dtype=np.float32) * 10.0
        kf.statePost = np.array([cx, cy, 0, 0], dtype=np.float32).reshape(4, 1)
        self.kf = kf

    def predict(self) -> Tuple[float, float]:
        s = self.kf.predict()
        return float(s[0, 0]), float(s[1, 0])

    def correct(self, cx: float, cy: float) -> Tuple[float, float]:
        measurement = np.array([[cx], [cy]], dtype=np.float32)
        s = self.kf.correct(measurement)
        # Clamp the corrected velocity in place so one bad association can't hand a huge
        # implied speed to every future (possibly unmatched/coasted) predict step.
        vx = float(np.clip(s[2, 0], -self.MAX_VELOCITY, self.MAX_VELOCITY))
        vy = float(np.clip(s[3, 0], -self.MAX_VELOCITY, self.MAX_VELOCITY))
        self.kf.statePost[2, 0] = vx
        self.kf.statePost[3, 0] = vy
        return float(s[0, 0]), float(s[1, 0])


@dataclass
class TrackState:
    """One frame's worth of a track's (smoothable) state."""
    frame_idx: int
    cx: float
    cy: float
    w: float
    h: float
    matched: bool  # True if a real detection was matched this frame (vs. Kalman-coasted)


@dataclass
class Track:
    track_id: int
    kf: KalmanPointTracker
    w: float           # box size, smoothed with a bounded EMA (see SIZE_EMA_ALPHA) -- NOT
    h: float            # part of the Kalman state, so it can't runaway (see class docstring)
    hits: int = 1        # real (matched) detections accumulated -- used for confirmation
    missed: int = 0        # consecutive frames coasted without a matching detection
    history: List[TrackState] = field(default_factory=list)


# How much a fresh matched detection nudges a track's remembered box size. Because this
# is a plain weighted average of the current size and a new measurement -- never a rate
# of change that gets extrapolated -- the result can never leave the range of sizes the
# detector itself has actually produced, no matter how many frames coast in between.
SIZE_EMA_ALPHA = 0.25

# A detection is only allowed to match an existing track if its area isn't wildly
# different from the track's current size. This is what stops a track from ever latching
# onto a much bigger (or smaller) region in the first place -- e.g. a stray blob picked
# up near a hand -- rather than cleaning up the damage after the fact.
SIZE_RATIO_MIN = 0.3
SIZE_RATIO_MAX = 3.0

# A last-resort hard ceiling applied only at render time. The EMA above already keeps
# tracked size within the detector's own output range, so this should never actually
# trigger -- it exists purely so a screen-filling box is structurally impossible even if
# some future change breaks that guarantee.
MAX_RENDER_BOX_SIDE = 120


def track_video(
    detections_per_frame: List[List[Detection]],
    max_distance: float = 25.0,
    max_missed: int = 8,
) -> Dict[int, Track]:
    """Track nuts across the whole clip with a per-track Kalman filter on position.

    Every track's complete history is kept (including frames before it accumulated
    enough hits to be trusted), because the confirmation/smoothing pass that follows
    needs the full trajectory to render a confirmed track from its very first frame.
    """
    active: Dict[int, Track] = {}
    all_tracks: Dict[int, Track] = {}
    next_id = 1
    max_dist_sq = max_distance ** 2

    for frame_idx, detections in enumerate(detections_per_frame):
        # Predict every active track first (this is what lets a track coast smoothly
        # through a frame the detector misses, instead of freezing or disappearing).
        predictions = {tid: track.kf.predict() for tid, track in active.items()}

        # Global greedy nearest-neighbour association by ascending predicted distance,
        # gated on BOTH proximity and size plausibility -- a nearby detection whose size
        # doesn't look like this track's nut is treated as no match at all.
        pairs = []
        for tid, (pcx, pcy) in predictions.items():
            track_area = max(active[tid].w * active[tid].h, 1.0)
            for di, det in enumerate(detections):
                _x, _y, dw, dh, (dcx, dcy) = det
                dist_sq = (pcx - dcx) ** 2 + (pcy - dcy) ** 2
                if dist_sq > max_dist_sq:
                    continue
                ratio = (dw * dh) / track_area
                if ratio < SIZE_RATIO_MIN or ratio > SIZE_RATIO_MAX:
                    continue
                pairs.append((dist_sq, tid, di))
        pairs.sort(key=lambda p: p[0])

        matched_tracks, matched_dets = set(), set()
        for _dist_sq, tid, di in pairs:
            if tid in matched_tracks or di in matched_dets:
                continue
            matched_tracks.add(tid)
            matched_dets.add(di)

            track = active[tid]
            _x, _y, w, h, (dcx, dcy) = detections[di]
            cx, cy = track.kf.correct(dcx, dcy)
            track.w = (1 - SIZE_EMA_ALPHA) * track.w + SIZE_EMA_ALPHA * float(w)
            track.h = (1 - SIZE_EMA_ALPHA) * track.h + SIZE_EMA_ALPHA * float(h)
            track.hits += 1
            track.missed = 0
            track.history.append(TrackState(frame_idx, cx, cy, track.w, track.h, True))

        for tid, track in active.items():
            if tid in matched_tracks:
                continue
            pcx, pcy = predictions[tid]
            track.missed += 1
            # Size is frozen (not extrapolated) while coasting -- see class docstring.
            track.history.append(TrackState(frame_idx, pcx, pcy, track.w, track.h, False))

        for di, det in enumerate(detections):
            if di in matched_dets:
                continue
            _x, _y, w, h, (cx, cy) = det
            kf = KalmanPointTracker(cx, cy)
            track = Track(track_id=next_id, kf=kf, w=float(w), h=float(h))
            track.history.append(TrackState(frame_idx, cx, cy, float(w), float(h), True))
            active[next_id] = track
            all_tracks[next_id] = track
            next_id += 1

        for tid in [tid for tid, t in active.items() if t.missed > max_missed]:
            del active[tid]

    return all_tracks


def smooth_track_history(history: List[TrackState], window: int = 9) -> List[TrackState]:
    """Zero-phase Gaussian-weighted moving-average smoothing of (cx, cy, w, h).

    Because the whole video is processed up front, this smooths each frame's state using
    BOTH past and future samples of the same track -- something a live/causal tracker
    can't do -- which removes residual shake left over after the Kalman filter.
    Edge padding (repeat first/last sample) is used so the box doesn't collapse or drift
    toward zero at the very start/end of a track's life.
    """
    n = len(history)
    if n < 3:
        return history

    window = max(3, min(window, n))
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return history

    half = window // 2
    sigma = window / 4.0
    offsets = np.arange(-half, half + 1)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()

    arr = np.array([[s.cx, s.cy, s.w, s.h] for s in history], dtype=np.float64)
    padded = np.pad(arr, ((half, half), (0, 0)), mode="edge")
    smoothed = np.zeros_like(arr)
    for k in range(window):
        smoothed += kernel[k] * padded[k:k + n]

    return [
        TrackState(
            history[i].frame_idx,
            float(smoothed[i, 0]),
            float(smoothed[i, 1]),
            float(smoothed[i, 2]),
            float(smoothed[i, 3]),
            history[i].matched,
        )
        for i in range(n)
    ]


def find_crossing_frame(history: List[TrackState], line_y: float, margin: float = 2.0):
    """First frame index where the (smoothed) trajectory crosses the line upward."""
    for i in range(1, len(history)):
        prev_y = history[i - 1].cy
        curr_y = history[i].cy
        if prev_y > line_y + margin and curr_y <= line_y + margin:
            return history[i].frame_idx
    return None


def run(
    input_path: str,
    output_path: str,
    line_y_ratio: float = 0.10,
    min_area: int = 55,
    max_area: int = 950,
    min_separation: int = 9,
    max_distance: float = 25.0,
    min_hits: int = 4,
    max_missed: int = 8,
    smooth_window: int = 9,
) -> Tuple[int, int]:
    """Process input video, detect and count nuts crossing the line, and save output video."""
    inp = Path(input_path)
    if not inp.exists():
        raise FileNotFoundError(f"Input video file does not exist: {input_path}")

    cap = cv2.VideoCapture(str(inp))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        line_y = int(height * line_y_ratio)

        # ---- Pass 1: run the tuned per-frame detector on every frame ----
        detections_per_frame: List[List[Detection]] = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            detections_per_frame.append(
                edge_watershed_detections(frame, min_area, max_area, min_separation)
            )
        total_frames = len(detections_per_frame)
    finally:
        cap.release()

    # ---- Pass 2: Kalman-filtered tracking across the whole clip ----
    all_tracks = track_video(detections_per_frame, max_distance, max_missed)

    # ---- Pass 3: confirm -- keep only tracks re-detected enough times to be real nuts,
    # not single-frame noise blips. Once confirmed, a track is rendered for its WHOLE
    # life (even the frames before it reached min_hits), which is only possible because
    # we're processing offline and can look at the future of the track before deciding.
    confirmed = {tid: t for tid, t in all_tracks.items() if t.hits >= min_hits}

    # ---- Pass 4 & 5: non-causal smoothing + robust line-crossing on the clean path ----
    frame_draws: Dict[int, List[Tuple[int, Tuple[float, float, float, float], Tuple[float, float]]]] = {
        i: [] for i in range(total_frames)
    }
    crossing_frame_by_track: Dict[int, int] = {}

    for tid, track in confirmed.items():
        smoothed = smooth_track_history(track.history, smooth_window)
        crossing = find_crossing_frame(smoothed, line_y)
        if crossing is not None:
            crossing_frame_by_track[tid] = crossing
        for s in smoothed:
            bbox = (s.cx - s.w / 2, s.cy - s.h / 2, s.w, s.h)
            frame_draws[s.frame_idx].append((tid, bbox, (s.cx, s.cy)))

    count_events = sorted(crossing_frame_by_track.items(), key=lambda kv: kv[1])
    final_count = len(count_events)
    # The label shown on a box should mean "the Nth nut counted so far" -- not the
    # internal track_id, which also gets consumed by short-lived tracks that never pass
    # confirmation (hand-region blips, etc.) and so races far ahead of the on-screen
    # COUNT. Assigning the display number by crossing order guarantees a track's number
    # can never exceed (or conflict with) the COUNT value shown at the same moment.
    display_number = {tid: idx + 1 for idx, (tid, _frame) in enumerate(count_events)}

    # ---- Pass 6: re-read the source video and render the stabilized overlay ----
    cap = cv2.VideoCapture(str(inp))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot re-open video for rendering: {input_path}")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create video writer for: {output_path}")

    try:
        counted_ids: set = set()
        event_iter = iter(count_events)
        next_event = next(event_iter, None)
        running_count = 0
        frame_index = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            while next_event is not None and next_event[1] <= frame_index:
                counted_ids.add(next_event[0])
                running_count += 1
                next_event = next(event_iter, None)

            for tid, (bx, by, bw, bh), (cx, cy) in frame_draws.get(frame_index, []):
                # Hard safety net -- see MAX_RENDER_BOX_SIDE definition.
                bw = min(bw, MAX_RENDER_BOX_SIDE)
                bh = min(bh, MAX_RENDER_BOX_SIDE)
                x, y, w, h = int(round(bx)), int(round(by)), int(round(bw)), int(round(bh))
                is_counted = tid in counted_ids
                color = (0, 220, 0) if is_counted else (0, 165, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.circle(frame, (int(round(cx)), int(round(cy))), 2, (0, 0, 255), -1)
                # Only counted (green) boxes get a number, and it's the nut's position in
                # the counting order -- so any number on screen always matches COUNT.
                # Not-yet-counted (orange) boxes are still tracked and drawn, just
                # unlabeled, since they don't have a count position yet.
                if is_counted:
                    cv2.putText(
                        frame,
                        f"nut {display_number[tid]}",
                        (x, max(15, y - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.38,
                        color,
                        1,
                        cv2.LINE_AA,
                    )

            # Draw UI annotations
            cv2.line(frame, (0, line_y), (width, line_y), (255, 0, 255), 3)
            cv2.putText(
                frame,
                "COUNTING LINE",
                (8, max(24, line_y - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.rectangle(frame, (7, 7), (205, 44), (20, 20, 20), -1)
            cv2.putText(
                frame,
                f"COUNT: {running_count}",
                (16, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.76,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            writer.write(frame)
            frame_index += 1
    finally:
        writer.release()
        cap.release()

    return final_count, total_frames


def main():
    script_dir = Path(__file__).parent.resolve()
    default_output = script_dir / "output" / "nuts_counted.mp4"

    p = argparse.ArgumentParser(
        description=(
            "Nut detection and counting using Canny edges & Watershed segmentation, "
            "stabilized with Kalman-filtered tracking and offline trajectory smoothing."
        )
    )
    p.add_argument(
        "--input",
        default=r"C:\Users\Ehab Walyaldeen\Documents\301 CV\WhatsApp Video 2023-04-25 at 10.00.20 AM.mp4",
        help="Path to input video file",
    )
    p.add_argument(
        "--output",
        default=str(default_output),
        help="Path to output video file",
    )
    p.add_argument(
        "--line-y-ratio",
        type=float,
        default=0.10,
        help="Vertical position ratio for counting line",
    )
    p.add_argument(
        "--min-area",
        type=int,
        default=55,
        help="Minimum area threshold for valid nut detections",
    )
    p.add_argument(
        "--max-area",
        type=int,
        default=950,
        help="Maximum area threshold for valid nut detections",
    )
    p.add_argument(
        "--min-separation",
        type=int,
        default=9,
        help="Min center-to-center px between two touching nuts' watershed peaks before they merge into one",
    )
    p.add_argument(
        "--max-distance",
        type=float,
        default=25.0,
        help="Max pixel distance to associate a detection with an existing track",
    )
    p.add_argument(
        "--min-hits",
        type=int,
        default=4,
        help="Detections a track needs before it's trusted as a real nut (filters one-off noise blips)",
    )
    p.add_argument(
        "--max-missed",
        type=int,
        default=8,
        help="Frames a track may coast on Kalman prediction alone before being dropped",
    )
    p.add_argument(
        "--smooth-window",
        type=int,
        default=9,
        help="Odd window size (in frames) for the non-causal trajectory smoothing pass",
    )
    args = p.parse_args()

    count, frames = run(
        args.input,
        args.output,
        args.line_y_ratio,
        args.min_area,
        args.max_area,
        args.min_separation,
        args.max_distance,
        args.min_hits,
        args.max_missed,
        args.smooth_window,
    )
    print(f"Processed {frames} frames. Final count: {count}. Output: {args.output}")


if __name__ == "__main__":
    main()
