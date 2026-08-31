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
