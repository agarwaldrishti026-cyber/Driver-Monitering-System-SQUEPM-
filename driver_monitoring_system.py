"""
=====================================================================================
DRIVER MONITORING SYSTEM (DMS)
Real-Time Drowsiness, Distraction and Fatigue Detection using OpenCV + MediaPipe
=====================================================================================

Features:
    - Real-time webcam monitorings
    - Face detection & tracking (MediaPipe FaceMesh)
    - Eye closure detection (Eye Aspect Ratio - EAR)
    - Drowsiness detection with configurable alarm timer
    - Head pose estimation (Yaw / Pitch / Roll) via solvePnP
    - Directional gaze alerts: LEFT / RIGHT / UP / DOWN sustained for too long
    - INSTANT head movement alert for fast, large-angle head turns
    - SUDDEN NECK MOVEMENT detection (large angle change between frames)
    - DISTRACTED DRIVER detection (repeated look-aways in a short window)
    - Multi-level (GREEN / YELLOW / ORANGE / RED) sustained head-pose warnings
    - Blink counter
    - Fatigue score (0-100) using PERCLOS + blink-rate + distraction history
    - FPS counter
    - Professional live dashboard overlay with head-direction timer
    - Audible alarm (non-blocking, thread based, cross-platform fallback)
    - Optional spoken voice alerts (pyttsx3, non-blocking, queued)
    - Status messages

    NEW in this version:
    - 3-second startup driver CALIBRATION (baseline yaw/pitch/EAR)
    - Stabilized head pose: short moving average + EMA + RELATIVE angle to the
      driver's own baseline + 2.0s sustain before the head-direction alert fires
    - Head pose is solved from RIGID facial landmarks only (eyes, nose bridge,
      nose tip, forehead, temples) against MediaPipe's canonical face model, so
      opening the mouth / moving the jaw can never be mistaken for a head turn
    - YAWN DETECTION via a pose-invariant 3D Mouth Aspect Ratio (MAR) plus an
      inner-lip opening check, confirmed only after the mouth stays genuinely
      open 2.0s continuously (ignores talking/smiling/head-down)
    - PHONE DISTRACTION detection (YOLOv8n, COCO "cell phone" class)
    - SEAT BELT detection (custom-trained YOLO model, optional/pluggable)
    - Automatic Trip_Report.pdf generation on exit
    - Optional Firebase Firestore cloud sync of each trip's stats

Run:
    pip install opencv-python mediapipe numpy
    pip install pyttsx3          (optional - enables spoken voice alerts)
    pip install ultralytics      (optional - enables phone / seat-belt detection)
    pip install reportlab        (optional - enables the Trip_Report.pdf)
    pip install firebase-admin   (optional - enables Firebase cloud sync)
    python driver_monitoring_system.py

Notes on optional features (all degrade gracefully if skipped):
    - Phone detection uses the stock "yolov8n.pt" model, which ultralytics
      auto-downloads on first run (COCO class 67 = "cell phone").
    - Seat-belt detection needs a CUSTOM-TRAINED YOLO model saved as
      "seatbelt_yolov8n.pt" next to this script (standard COCO models have no
      seat-belt class). If that file isn't present, this feature simply stays
      off - everything else keeps working.
    - Firebase sync needs a service-account JSON key - see the FirebaseManager
      class docstring below for the exact setup steps.

Controls:
    q / ESC  -> Quit
    r        -> Reset session statistics
=====================================================================================
"""

import os
import cv2
import time
import math
import statistics
import threading
import collections
import sys
import urllib.request

import numpy as np
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------------
# Optional platform-specific beep support (graceful fallback on non-Windows OS)
# ---------------------------------------------------------------------------------
try:
    import winsound
    _HAS_WINSOUND = True
except ImportError:
    _HAS_WINSOUND = False

# ---------------------------------------------------------------------------------
# Optional voice-alert support (graceful fallback if pyttsx3 isn't installed)
# ---------------------------------------------------------------------------------
try:
    import pyttsx3
    _HAS_PYTTSX3 = True
except ImportError:
    _HAS_PYTTSX3 = False


# =====================================================================================
# FIX (audio audibility, req #3/#4): ensure_alert_wav()
# =====================================================================================
# Synthesizes a short, attention-getting two-tone alert WAV file using ONLY the
# Python standard library ('wave' + 'struct' - no numpy/scipy/extra pip installs,
# no external audio file to download). Only writes the file if it does not
# already exist, so this costs nothing on every subsequent run. This is what
# AlarmManager plays via winsound.PlaySound() for a reliably audible alert.
# =====================================================================================
def ensure_alert_wav(path):
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return path, None
    try:
        import wave
        import struct
        import math as _math

        sample_rate = 44100
        # Two short, distinct urgent tones back-to-back (~500ms total) - more
        # attention-getting than a single flat tone and easy to tell apart
        # from ordinary notification sounds.
        segments = [(1000, 0.18), (750, 0.18)]
        samples = []
        fade_samples = max(1, int(sample_rate * 0.01))  # 10ms fade in/out per segment, avoids clicks
        for freq, duration in segments:
            n = int(sample_rate * duration)
            for i in range(n):
                t = i / sample_rate
                envelope = min(1.0, i / fade_samples, (n - i) / fade_samples)
                value = _math.sin(2 * _math.pi * freq * t) * envelope
                samples.append(int(max(-1.0, min(1.0, value)) * 32767 * 0.6))

        with wave.open(path, 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(struct.pack('<%dh' % len(samples), *samples))
        return path, None
    except Exception as exc:
        # FIX (req #13): never silently swallow this - the caller (AlarmManager /
        # startup diagnostics) prints the real reason WAV playback isn't available.
        return None, str(exc)


# =====================================================================================
# CONFIGURATION
# =====================================================================================
class Config:
    # Camera
    CAMERA_INDEX = 0
    FRAME_WIDTH = 960
    FRAME_HEIGHT = 540

    # EAR (Eye Aspect Ratio)
    EAR_THRESHOLD = 0.21          # Below this -> eye considered closed
    # FIX (Phase 1, EAR): time-based (not frame-count) minimum closure duration
    # to confirm a real blink vs single-frame EAR noise. Frame counts silently
    # change meaning with FPS (2 frames = 133ms @15fps but 33ms @60fps); a
    # fixed time threshold behaves identically regardless of camera/pipeline FPS.
    EAR_CONSEC_MIN_SECONDS = 0.06   # ~2 frames at 30fps, but FPS-independent

    # Drowsiness (sustained eye closure) - PROBLEM 5: alarm now only fires at
    # the CRITICAL stage (>= 4.0s). See EYE_STAGE_* constants further below
    # for the full WARNING -> FATIGUE -> DROWSY -> CRITICAL staging.
    EYE_CLOSED_ALARM_SECONDS = 4.0    # Eyes closed continuously -> CRITICAL/alarm

    # NOTE: gaze direction is classified using RELATIVE, calibrated, smoothed
    # angles (see the "REDESIGNED: automotive-grade head-pose logic" block
    # below) rather than fixed absolute camera angles, since a driver's
    # neutral pose varies with seat position / camera mount.

    # ---- NEW: sudden neck-movement detection ------------------------------------
    SUDDEN_TURN_MIN_DELTA_DEG = 20.0     # min angle change between frames
    SUDDEN_TURN_MAX_TIME_SEC = 0.35      # within this time window
    SUDDEN_TURN_COOLDOWN_SEC = 1.5       # avoid alert spam

    # ---- NEW: distracted-driver detection (frequency based) --------------------
    DISTRACTION_EVENT_LIMIT = 5
    DISTRACTION_WINDOW_SECONDS = 20.0

    # NOTE: face-loss handling (Issue 1) freezes ALL detection/alerts and shows
    # "FACE NOT DETECTED" / "STABILIZING..." directly - see FACE_STABILITY_SECONDS
    # below and the face-gating block in run(). There is no separate no-face
    # alarm timer anymore; that would fire a voice/alarm alert on face loss,
    # which is explicitly not wanted.

    # ---- NEW: voice alerts -------------------------------------------------------
    ENABLE_VOICE_ALERTS = True
    VOICE_COOLDOWN_SECONDS = 4.0

    # ---- NEW: face-detection gating -----------------------------------------------
    # While no face (or a just-reacquired, not-yet-stable face) is present, ALL
    # detections (EAR/MAR/pose/blink/fatigue) and ALL alerts (voice + sustained
    # head/eye/yawn timers) are frozen. Detection only resumes once the face has
    # been continuously present for FACE_STABILITY_SECONDS.
    FACE_STABILITY_SECONDS = 1.0

    # ---- REDESIGNED: automotive-grade head-pose logic -------------------------------
    # Angles are always RELATIVE to the driver's own calibrated baseline (not raw
    # camera angle) and smoothed over POSE_SMOOTHING_FRAMES frames. Four bands,
    # with a hysteresis gap between "ignore" and "start monitoring" so ordinary
    # driving movements (mirror checks, etc.) never trigger anything:
    #
    #   0        -> REL_DEADZONE_DEG        : ignored completely (camera/landmark noise)
    #   REL_DEADZONE_DEG -> REL_MONITOR_START_DEG : hysteresis gap (still ignored)
    #   REL_MONITOR_START_DEG -> REL_WARNING_DEG  : "monitoring" (duration tracked, no alert)
    #   >= REL_WARNING_DEG, sustained HEAD_WARNING_SUSTAIN_SECONDS+ : WARNING
    #   >= REL_INSTANT_DEG (any duration, debounced a few frames)   : INSTANT ALERT
    # FIX (alert latency): was 15 frames (SMA) + EMA alpha 0.25, i.e. ~0.5-0.8 s of
    # pure smoothing lag BEFORE the sustain timer could even start. The pose is now
    # solved from rigid landmarks only (much less jitter), and the deadzone, the
    # hysteresis latch, the confidence gate and the sustain timer already reject
    # ordinary movement, so a short window is enough.
    POSE_SMOOTHING_FRAMES = 8           # moving-average window for yaw/pitch/roll
    # FIX (axes/signs, verified numerically): the pose is now solved against a
    # model that uses the SAME axis convention as the camera (x right, y down,
    # z away - see MODEL_POINTS_3D), and yaw/pitch are read from the face's
    # forward vector, so no axis remapping is needed any more. These were
    # checked against synthetic faces projected through a camera model:
    # +yaw = nose moves toward the RIGHT of the (mirrored) image, +pitch = nose
    # moves UP. Both are therefore +1.0. Flip one to -1.0 only if a particular
    # camera/mount ever reports UP/DOWN or LEFT/RIGHT mirrored.
    HEAD_POSE_PITCH_SIGN = 1.0           # +pitch must mean UP (nose tilts up)
    HEAD_POSE_YAW_SIGN = 1.0             # +yaw must mean RIGHT turn
    REL_DEADZONE_DEG = 20.0             # below this -> completely ignored (normal driving)
    REL_MONITOR_START_DEG = 25.0        # duration-tracking starts beyond this
    REL_WARNING_DEG = 35.0              # must stay beyond this to ever warn
    REL_INSTANT_DEG = 55.0              # instant alert threshold (extreme turn)
    # FIX (alert timing): the head-direction ALERT (RED banner + alarm + voice)
    # now fires after 2.0 s. Previously 2.5 s only produced a silent YELLOW
    # label; the audible alert did not arrive until HEAD_RED_SUSTAIN_SECONDS
    # (5.0 s), which is why it felt so slow. YELLOW / ORANGE are now early,
    # VISUAL-ONLY stages that lead up to the 2.0 s alert instead of following it.
    HEAD_YELLOW_SUSTAIN_SECONDS = 1.0   # early notice (visual only)
    HEAD_ORANGE_SUSTAIN_SECONDS = 1.5   # stronger notice (visual only, no voice)
    HEAD_WARNING_SUSTAIN_SECONDS = 2.0  # beyond the warning angle this long -> ALERT
    HEAD_RED_SUSTAIN_SECONDS = HEAD_WARNING_SUSTAIN_SECONDS   # same trigger as above
    HEAD_LOW_LIGHT_SUSTAIN_FACTOR = 1.0 # multiplies ALL head sustain times in low
                                         # light (was a hard-coded 1.4, which pushed
                                         # the first alert to ~3.5 s). Raise it if
                                         # dim cabins give too many false alerts.
    HEAD_DIRECTION_GRACE_SECONDS = 0.3  # a momentary dip below the monitoring angle
                                         # (landmark jitter, one failed pose frame)
                                         # does not restart the sustain timer
    HEAD_HYSTERESIS_RESET_DEG = 20.0    # must fall back below this before the SAME
                                         # direction is allowed to warn/alert again
                                         # (prevents alerts repeatedly re-triggering
                                         # while oscillating around the threshold)
    INSTANT_ALERT_CONSEC_FRAMES = 3     # require the extreme angle to hold for this
                                         # many consecutive RAW frames (never camera noise)

    # ---- NEW (industry-grade tuning): direction-SPECIFIC thresholds ------------
    # Real ADAS systems tolerate more yaw (LEFT/RIGHT, e.g. mirror checks) than
    # pitch (UP/DOWN nods) before warning - mirror checks are frequent/normal,
    # sustained up/down looks are not. These are the per-axis "recommended"
    # (WARNING) and "emergency" (INSTANT) angles in degrees, relative to the
    # driver's calibrated baseline:
    #   Yaw   recommended = +-30 deg   |  Yaw   emergency = +-45 deg
    #   Pitch recommended = +-22 deg   |  Pitch emergency = +-30 deg
    # REL_DEADZONE_DEG / REL_MONITOR_START_DEG / REL_WARNING_DEG / REL_INSTANT_DEG
    # above remain as general fallbacks (still used for smoothing/back-compat);
    # the ratios below derive each axis's own deadzone/monitor-start band from
    # its WARNING angle, preserving the same hysteresis-gap shape.
    # PROBLEM 1 SPEC: LEFT/RIGHT > 35 deg, UP/DOWN > 25 deg, sustained
    # continuously for >= HEAD_WARNING_SUSTAIN_SECONDS (2.5s) before any alert -
    # if the driver returns to normal before that, the alert is cancelled
    # (see the hysteresis-reset / direction_start_time=None logic below).
    #
    # FIX (head-pose RED bug): lowered from 35/25 deg. A natural, deliberate
    # "look LEFT/RIGHT/UP/DOWN and hold" head turn (e.g. checking a mirror,
    # glancing at a passenger, looking down at a phone) very often falls in
    # the 20-30 deg range and was crossing into "monitoring" (yellow
    # indicator, via AXIS_MONITOR_START_RATIO below) without ever reaching
    # the old 35/25 deg WARNING angle itself - so reached_warning_deg in
    # process_head_direction() stayed False forever, duration kept
    # accumulating but level was hard-forced to GREEN regardless of how long
    # it was held, and RED (+ the audible alert) was never reached even
    # though the driver was visibly, sustainedly turned away. These values
    # now match common real-world DMS practice for a genuine warning angle.
    YAW_WARNING_DEG = 28.0
    YAW_INSTANT_DEG = 50.0
    PITCH_WARNING_DEG = 18.0
    PITCH_INSTANT_DEG = 35.0
    AXIS_DEADZONE_RATIO = 0.55         # below this fraction of WARNING_DEG -> fully ignored
    AXIS_MONITOR_START_RATIO = 0.72    # above this fraction -> "monitoring" (no alert yet)
    AXIS_HYSTERESIS_RESET_RATIO = 0.55 # must fall back below this fraction to re-arm

    # FIX (head-pose RED bug, part 2): pose_confidence used to be read as a
    # single instantaneous per-frame value gating ALL escalation - one noisy
    # frame (common with webcam landmark jitter, especially at larger yaw/
    # pitch angles where MediaPipe's own tracking quality naturally dips a
    # little) could momentarily drop it under MIN_POSE_CONFIDENCE and force
    # level back to GREEN even mid-way through an otherwise clearly
    # sustained, deliberate head turn. It is now averaged over a short
    # rolling window (see self._pose_confidence_hist) so a single bad frame
    # can no longer erase several seconds of consistent tracking.
    POSE_CONFIDENCE_SMOOTHING_FRAMES = 8

    # ---- PROBLEM 9: continuous single-glance distraction (separate from the
    # frequency-based DISTRACTION_EVENT_LIMIT counter above) - a look that
    # itself stays past the warning angle continuously for this long is
    # "DISTRACTED DRIVER" on its own; short glances never reach it.
    DISTRACTION_CONTINUOUS_SECONDS = 3.0

    # ---- NEW: exponential moving average (extra jitter suppression) ------------
    # Layered on TOP of the existing POSE_SMOOTHING_FRAMES simple moving average:
    # the SMA removes high-frequency camera/landmark noise over its window, and
    # this EMA further removes any residual frame-to-frame jitter/shake without
    # adding perceptible extra lag (Feature 9/11: "no jitter", "use EMA").
    POSE_EMA_ALPHA = 0.5   # (was 0.25 - see POSE_SMOOTHING_FRAMES for why lag was cut)

    # ---- NEW: confidence filtering (Feature 6) ----------------------------------
    # This is OUR downstream gating bar: the jitter-based _pose_confidence()/
    # FaceStateMachine.confidence() score must clear this before any head-pose
    # WARNING/INSTANT alert can fire.
    MIN_POSE_CONFIDENCE = 0.7

    # FIX (Phase 1, goal #1 root cause): MediaPipe's OWN internal detection vs
    # tracking confidence are now DIFFERENT values instead of both being hard-
    # coded to MIN_POSE_CONFIDENCE. Using the same (high, 0.7) value for both
    # made initial face ACQUISITION unnecessarily strict, contributing to the
    # "face detection feels flaky" symptom. Detection confidence is relaxed so
    # a face is picked up readily; tracking confidence is only mildly relaxed
    # from before, since our own FaceStateMachine debounce (not MediaPipe's
    # internal confidence) is now what protects against transient flicker.
    MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.5
    MEDIAPIPE_MIN_TRACKING_CONFIDENCE = 0.6

    # ---- NEW: unified alert cooldown (Feature 8) --------------------------------
    # The SAME alert (matched by message text) cannot repeat within this many
    # seconds - a different, more urgent message is never blocked by it though.
    ALERT_COOLDOWN_SECONDS = 5.0

    # ---- NEW: staged face-lost escalation (Feature 5) ---------------------------
    FACE_LOST_SEARCHING_SECONDS = 2.0   # 0-2s -> "Searching..." (GREEN, no alert)
    FACE_LOST_WARNING_SECONDS = 5.0     # 2-5s -> WARNING (YELLOW); 5s+ -> DANGER (RED)

    # ---- NEW (Phase 1 - unified face state machine) -----------------------------
    # Single set of debounce/ramp constants used by FaceStateMachine (the ONE
    # source of truth for face presence/stability/occlusion - see that class
    # for the full design rationale). Frame-based debounce counts are
    # intentionally frame-based (not time-based): they exist purely to ignore
    # single-frame MediaPipe blips (blinks, motion blur) regardless of FPS.
    FACE_MISS_DEBOUNCE_FRAMES = 3   # consecutive missed frames before "genuinely absent"
    FACE_HIT_DEBOUNCE_FRAMES = 2    # consecutive present frames before "genuinely reacquired"
    OCCLUSION_CONFIRM_FRAMES = 4    # consecutive bad-geometry frames before OCCLUDED
                                     # (occlusion detector itself is wired in a later Phase 1 fix;
                                     # the state machine already supports it so no rewiring later)
    OCCLUSION_CLEAR_FRAMES = 3      # consecutive clean-geometry frames before back to TRACKING
    CONFIDENCE_RAMP_FRAMES = 10     # frames after entering TRACKING before confidence is fully
                                     # trusted (stops instant-full-confidence right after reacquiring)

    # ---- NEW (Phase 1, goal #7): geometric occlusion thresholds ------------------
    # All ratios are relative to the tracked face bbox size, so they are scale-
    # (distance-to-camera-) invariant. Thresholds are intentionally conservative
    # (i.e. low) - a genuine 90-degree profile turn already shrinks these ratios
    # a lot with zero occlusion involved, so this must only fire on geometry that
    # is clearly degenerate, not on ordinary head turns. This keeps the detector
    # high-precision (few false "occluded" alerts) even though it trades away
    # some recall - appropriate for goal #10 (remove false alerts).
    OCCLUSION_MIN_INTER_OCULAR_RATIO = 0.08   # eye-corner-to-eye-corner distance / bbox width
    OCCLUSION_MIN_MOUTH_WIDTH_RATIO = 0.03    # mouth-corner-to-corner distance / bbox width
    OCCLUSION_MIN_EYE_MOUTH_RATIO = 0.15      # eye-midpoint-to-mouth-midpoint distance / bbox height

    # ---- NEW (Phase 1, goal #9): safety score rolling window ---------------------
    # How far back "recent behavior" looks for the LIVE safety score. Fixes the
    # root cause where yawn/phone/head-alert counts used to be lifetime-session
    # totals that only ever grew, so the score could never recover even after
    # a long stretch of good driving. distraction_events already used a rolling
    # window (DISTRACTION_WINDOW_SECONDS) and is left as-is.
    SAFETY_SCORE_WINDOW_SECONDS = 300.0   # 5 minutes

    EAR_CALIBRATION_MARGIN = 0.05      # (legacy fallback, kept for reference -
                                        # superseded by EAR_CLOSED_THRESHOLD_RATIO below)
    # PROBLEM 3 SPEC: measure the driver's natural (open-eye) EAR during the
    # first CALIBRATION_SECONDS (5s) of the session, then derive the
    # closed-eye threshold as Open_EAR * EAR_CLOSED_THRESHOLD_RATIO. This
    # scales correctly for every face shape instead of using one fixed value.
    CALIBRATION_SECONDS = 5.0          # length of the startup calibration phase
    EAR_CLOSED_THRESHOLD_RATIO = 0.72  # dynamic closed threshold = open EAR * 0.72

    # ---- FIX (calibration bug): outlier rejection during calibration -----------
    # A sample farther than this from the MEDIAN of all calibration samples is
    # excluded before averaging the final baseline (see run_calibration_step's
    # robust_baseline()). Prevents a few transient/noisy frames (driver still
    # settling into position, a blink, motion blur) from dragging the baseline
    # to an implausible value like the reported "Baseline pitch=-110.9".
    CALIBRATION_OUTLIER_REJECT_DEG = 25.0    # for yaw/pitch samples (degrees)
    CALIBRATION_OUTLIER_REJECT_EAR = 0.08    # for EAR/MAR samples (ratio units)

    # ---- PROBLEM 4: blink vs. drowsiness duration cutoff ------------------------
    # A normal blink is ~100-300ms. Anything closed continuously longer than
    # this is NOT a blink - it's counted toward drowsiness/fatigue instead.
    BLINK_MAX_DURATION_SECONDS = 0.6

    # ---- PROBLEM 5: staged drowsiness escalation (eyes continuously closed) ----
    #   0.0 - 1.0s  -> WARNING   (early notice, no alert)
    #   1.0 - 2.5s  -> FATIGUE   (dashboard/YELLOW)
    #   2.5 - 4.0s  -> DROWSY    (ORANGE)
    #   >= 4.0s     -> CRITICAL  (RED + audible alarm - only stage that alarms)
    EYE_STAGE_WARNING_SECONDS = 1.0
    EYE_STAGE_FATIGUE_SECONDS = 2.5
    EYE_STAGE_DROWSY_SECONDS = 4.0
    # kept for any other code/dashboard text still referencing these names:
    DROWSY_WARNING_SECONDS = EYE_STAGE_WARNING_SECONDS

    # ---- NEW (Feature 2): yawn detection (Mouth Aspect Ratio) -------------------
    MAR_THRESHOLD = 0.55               # fallback above this -> mouth "open"
                                        # (overridden by the calibrated dynamic
                                        # threshold once calibration completes)
    MAR_CALIBRATION_MARGIN = 0.30      # added to baseline (relaxed, neutral-mouth)
                                        # MAR to get the dynamic open-mouth threshold.
                                        # (was 0.20: the MAR is now a pose-invariant 3D
                                        # ratio, and a larger margin keeps talking /
                                        # a slightly parted mouth from counting as a
                                        # yawn now that the confirmation time is 2.0 s)
    # FIX: an inner-lip opening check that must ALSO hold. The MAR above mixes
    # outer-lip distance (present even with a closed mouth) with inner-lip gap;
    # this requires the lips themselves to be clearly parted (inner gap divided
    # by mouth width), which is what a real yawn does and what a lowered head or
    # a jittery lower-lip landmark does not.
    YAWN_INNER_MIN_RATIO = 0.40
    YAWN_MIN_DURATION_SECONDS = 2.0    # mouth must stay open continuously this
                                        # long to confirm a YAWN (ignores talking/
                                        # smiling/singing/short mouth movements)
    YAWN_MAR_SMOOTHING_FRAMES = 5

    # FIX (yawn reliability): a real yawn's MAR trace is rarely a perfectly
    # smooth monotonic rise - landmark jitter can cause it to briefly dip
    # back under the threshold for a frame or two mid-yawn. Previously any
    # single dip immediately reset mouth_open_start_time to None, restarting
    # the YAWN_MIN_DURATION_SECONDS timer from zero - so a genuine ~5s yawn
    # with one brief mid-yawn dip could reset itself repeatedly and never
    # reach confirmation. A short grace period now tolerates brief dips
    # without resetting the timer, while a dip longer than this still
    # correctly resets it (so talking, which has much longer closed-mouth
    # gaps between words, is still rejected).
    # FIX: 0.35 s was long enough to bridge the gaps between wide spoken syllables
    # (~0.2 s), so continuous talking could add up to a "yawn" once the timer was
    # shortened to 2.0 s (reproduced in simulation). 0.15 s (~4 frames) still
    # absorbs landmark jitter but not the pauses of speech.
    YAWN_DIP_GRACE_SECONDS = 0.15

    # ---- NEW (Feature 4): phone-distraction detection (YOLO) --------------------
    ENABLE_PHONE_DETECTION = True
    # Pretrained Ultralytics model (COCO). COCO class 67 = "cell phone".
    # Auto-downloaded by ultralytics on first run - no manual placeholder needed.
    YOLO_PHONE_MODEL_PATH = "yolov8n.pt"
    PHONE_CONF_THRESHOLD = 0.45
    PHONE_ALERT_SECONDS = 2.0          # phone near face this long -> alert

    # ---- FIX (phone-to-ear / calling detection): separate, tighter layer on
    # top of the existing phone-near-face detection above. A phone held up
    # in front of the face (texting/photo) should be PHONE DETECTED only; a
    # phone held to either ear should additionally raise PHONE CALLING.
    # FIX (phone-to-ear reliability / false calls):
    #  * PHONE_EAR_PROXIMITY_SCALE is now a fraction of the FACE WIDTH (ear-to-ear
    #    distance, floored for turned heads), and it is measured from the ear
    #    point to the nearest EDGE of the phone box. It used to be 0.85 x the
    #    larger face-box side measured to the box CENTER, which was so large that
    #    a phone held straight in front of the face also counted as "at the ear".
    #  * PHONE_EAR_LATERAL_MIN: the phone box centre must additionally sit on the
    #    outer side of that ear (fraction of the half face-width from the face
    #    centre). A phone in front of the face sits near 0; one at an ear is >1.
    PHONE_EAR_PROXIMITY_SCALE = 0.45
    PHONE_EAR_LATERAL_MIN = 0.70
    PHONE_CALL_ALERT_SECONDS = 2.0      # phone near ear this long -> PHONE CALLING
    # A phone at the ear is partly hidden by the hand/head, so YOLO scores it
    # lower than a phone held in the open. ONE inference is run at this lower
    # confidence; only boxes that also pass the ear-region geometry may use the
    # lower scores. Near-face logic still requires PHONE_CONF_THRESHOLD.
    PHONE_EAR_CONF_THRESHOLD = 0.30
    # YOLO output flickers frame to frame; both phone timers survive a gap this
    # short instead of restarting from zero (otherwise a continuous 2.0 s was
    # almost impossible to reach). Also how long the last boxes are re-used.
    PHONE_LOST_GRACE_SECONDS = 0.7
    YOLO_DETECT_EVERY_N_FRAMES = 3     # run YOLO every Nth frame (perf); shared
                                        # cadence for both phone + seat-belt models,
                                        # previous detections are reused in between
                                        # (was 5; raise it if your FPS drops)
    YOLO_INFER_MAX_SIDE = 640          # downscale the longest frame side to this
                                        # many pixels before YOLO inference (perf).
                                        # (was 480; ultralytics letterboxes to 640
                                        # anyway, so 480 only threw detail away -
                                        # small phones at the ear lost resolution)
    PHONE_FACE_PROXIMITY_SCALE = 0.6   # the gap between the phone box and the face box
                                        # must be within this multiple of the face-box
                                        # size to count as "near face" (keeps out
                                        # passenger/background phones). Was 1.8 measured
                                        # centre-to-centre, which covered most of the frame.
    PHONE_MIN_BOX_AREA_FRACTION = 0.001  # ignore tiny/far-away phone detections
                                          # (fraction of total frame area)

    # ---- NEW (Feature 5): seat-belt detection (YOLO) ----------------------------
    # NOTE: standard COCO-pretrained YOLO models do NOT include a "seat belt"
    # class, so this requires a custom-trained model. If SEATBELT_MODEL_PATH
    # does not exist on disk, seat-belt detection is automatically disabled
    # (the rest of the system keeps working normally). See the README notes
    # printed at startup for how to train/plug in such a model.
    ENABLE_SEATBELT_DETECTION = True
    YOLO_SEATBELT_MODEL_PATH = "seatbelt_yolov8n.pt"
    SEATBELT_CONF_THRESHOLD = 0.5
    SEATBELT_MISSING_ALERT_SECONDS = 2.0

    # ---- NEW (Feature 6): trip report (PDF) -------------------------------------
    ENABLE_TRIP_REPORT = True
    TRIP_REPORT_PATH = "Trip_Report.pdf"

    # ---- NEW (Feature 7): cloud sync (Firebase Firestore) -----------------------
    ENABLE_FIREBASE = False   # flip to True once you've done the setup below
    FIREBASE_CREDENTIALS_JSON = "firebase_credentials.json"
    FIREBASE_COLLECTION = "trip_reports"
    DRIVER_NAME = "Unknown Driver"

    # Fatigue scoring window
    PERCLOS_WINDOW_SECONDS = 60.0
    BLINK_RATE_WINDOW_SECONDS = 60.0
    NORMAL_BLINKS_PER_MIN = 17.0
    # FIX (false drowsiness-while-eyes-open bug): don't trust the blink-RATE
    # signal until this much real session time has passed - see the fusion
    # block in run() for the full explanation of why an early low reading is
    # expected/meaningless, not a sign of fatigue.
    MIN_SESSION_SECONDS_FOR_BLINK_SIGNAL = 30.0
    # NEW: smooth fatigue-score recovery (EMA) so the number doesn't jump/snap
    # around; higher alpha = trusts the new instantaneous reading more.
    FATIGUE_SMOOTHING_ALPHA = 0.15

    # ---- PROBLEM 6: Safety Score must update smoothly, never jump by 20pts ----
    # in a single frame - the raw event-based score is passed through this EMA
    # before it's ever shown or reported.
    SAFETY_SCORE_SMOOTHING_ALPHA = 0.10
    SAFETY_SCORE_MAX_STEP_PER_FRAME = 3.0   # hard cap, belt-and-braces on top of the EMA

    # ---- PROBLEM 10: EMA smoothing for EAR (on top of the existing short
    # moving-average window) to eliminate residual flicker, mirroring the
    # yaw/pitch/roll EMA already used for head pose.
    EAR_EMA_ALPHA = 0.35

    # Alarm
    BEEP_FREQ = 2200
    BEEP_DURATION_MS = 350

    # ---- FIX (audio audibility): WAV-based alarm sound ---------------------------
    # winsound.Beep() synthesizes a tone through a very basic/limited path that is
    # frequently much quieter (or effectively inaudible) than normal application
    # audio on modern Windows systems - it does not always route through the same
    # output device/volume mixer as everything else. Playing an actual WAV file
    # through winsound.PlaySound() uses the standard Windows audio mixer, exactly
    # like any other application sound, and is far more reliably audible. The WAV
    # is synthesized automatically on first run (see ensure_alert_wav()) - no
    # manual audio file download required.
    ALERT_WAV_FILENAME = "alert_beep.wav"
    ALARM_REPEAT_COOLDOWN_SECONDS = 1.2   # gap between repeats while ACTIVE
    ALERT_SOUND_COOLDOWN_SECONDS = 3.0    # min gap between "[Alert] X -> SOUND"
                                            # diagnostic prints, per alert type

    # ---- NEW (debug diagnostics, req #5): optional console status lines for
    # head/phone/yawn state, for debugging only - never replaces the dashboard.
    DEBUG_CONSOLE_DIAGNOSTICS = True
    DEBUG_CONSOLE_INTERVAL_SECONDS = 1.0   # throttle so it doesn't spam every frame

    # ---- NEW: robustness / error handling ---------------------------------------
    CAMERA_READ_FAIL_RECONNECT_ATTEMPTS = 5   # consecutive failed reads before
                                                # trying to fully reopen the camera
    CAMERA_READ_FAIL_ABORT_LIMIT = 60          # consecutive failed reads before
                                                # giving up entirely
    LOW_LIGHT_BRIGHTNESS_THRESHOLD = 55.0      # mean grayscale brightness (0-255)
                                                # below which we flag "LOW LIGHT"
    ENABLE_LOW_LIGHT_ENHANCEMENT = True         # apply CLAHE contrast boost when dim

    # UI
    # ---- Console layout (Part 2D / Part 3): a compact live camera feed on
    # the LEFT and a large, detailed Driver Safety Monitor dashboard on the
    # RIGHT. These two constants define the size of the final composed
    # window - the raw camera frame is resized DOWN to VIDEO_DISPLAY_WIDTH
    # so it never dominates the screen, and the dashboard gets a wide,
    # dedicated (non-overlapping) panel instead of a thin HUD stapled onto
    # the camera image.
    VIDEO_DISPLAY_WIDTH = 480      # compact camera column width (px)  (was 460)
    PANEL_WIDTH = 660              # large dashboard column width (px)  (was 620)
    CANVAS_HEIGHT = 800            # total composed window height (px)  (was 720)
    FONT = cv2.FONT_HERSHEY_SIMPLEX


# =====================================================================================
# NEW (Phase 1): UNIFIED FACE STATE MACHINE
# =====================================================================================
class FaceState:
    """
    Enumeration of the driver's face-tracking state. This is the SINGLE
    source of truth consumed by every other module (EAR, head pose,
    confidence, safety score, alerts, dashboard). No other part of the
    system is allowed to independently decide whether the face is
    "present" / "stable" / "lost" - see FaceStateMachine below.
    """
    SEARCHING = "SEARCHING"       # no face yet, but only briefly (grace period)
    STABILIZING = "STABILIZING"   # face just (re)acquired, not yet trusted
    TRACKING = "TRACKING"         # face present, stable, geometry sane -> full detection active
    OCCLUDED = "OCCLUDED"         # face present but partially covered (hand/object over face)
    FACE_LOST = "FACE_LOST"       # face genuinely absent beyond the search grace period


class FaceStateMachine:
    """
    PRODUCTION FIX (Phase 1 - unified face state):
    Single source of truth for face presence / stability / occlusion for the
    ENTIRE project. Every other module reads `.state` / `.is_tracking` /
    `.confidence(...)` from ONE instance of this class instead of maintaining
    its own face_present / face_stable / no_face_start_time /
    face_first_seen_time bookkeeping - which is exactly what caused
    conflicting states like "Face Detected: NO" while landmarks existed,
    "Searching..." while the face was visible, or 0% confidence while
    actively tracking.

    Design:
      - Frame-level presence/occlusion flags are DEBOUNCED (must persist for
        several consecutive frames) before the state changes, so a single
        dropped MediaPipe frame (blink, motion blur, brief light flicker)
        never cascades into a false "face lost" event or a restarted
        stability timer. Debounce is frame-COUNT based (not time based) so
        it behaves identically regardless of the current FPS.
      - Confidence is a pure function of state: exactly 0.0 unless TRACKING,
        and ramps up over CONFIDENCE_RAMP_FRAMES once TRACKING begins
        (instead of optimistically defaulting to full confidence the instant
        a face reappears), so head-pose/EAR alerts cannot fire the moment a
        face is reacquired.
      - update() is called exactly ONCE per frame from run() and is the ONLY
        place in the whole project allowed to change face state. Every other
        function receives the already-decided state/booleans - no module
        re-derives its own version of "is the face okay right now".
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.state = FaceState.SEARCHING

        # Debounce counters (frame counts, not seconds - see class docstring)
        self._present_streak = 0
        self._absent_streak = 0
        self._occluded_streak = 0
        self._clear_streak = 0

        # Timing bookkeeping - the ONLY copies of these timestamps in the
        # whole project. Nothing else should keep a parallel "first seen" /
        # "no face since" timestamp.
        self._state_entered_at = time.time()
        self._absence_started_at = None     # when we first left TRACKING/STABILIZING
        self._tracking_entered_at = None    # drives the confidence ramp

    # --------------------------------------------------------------------
    def reset(self):
        """Full reset - called from DriverMonitoringSystem.reset_session()."""
        self.__init__(self.cfg)

    # --------------------------------------------------------------------
    def update(self, landmarks_present, occlusion_detected, now):
        """
        Advance the state machine by exactly one frame. Pure state-transition
        logic only - no drawing, no alert text, no EAR/pose math lives here.

        Args:
            landmarks_present: True if MediaPipe returned a face this frame.
            occlusion_detected: True if a geometric check flags the visible
                face as partially covered. Always pass False until the
                occlusion detector (a later Phase 1 fix) is wired in - this
                state machine already supports it so nothing here will need
                to change when that lands.
            now: current timestamp (time.time()).

        Returns:
            True if a state transition just happened this frame (useful for
            callers that need to reset transition-triggered bookkeeping,
            e.g. clearing smoothing buffers exactly once on TRACKING exit -
            never on every individual missed frame).
        """
        prev_state = self.state

        # ---- update raw debounce streaks --------------------------------
        if landmarks_present:
            self._present_streak += 1
            self._absent_streak = 0
        else:
            self._absent_streak += 1
            self._present_streak = 0

        if occlusion_detected:
            self._occluded_streak += 1
            self._clear_streak = 0
        else:
            self._clear_streak += 1
            self._occluded_streak = 0

        # ---- transitions -------------------------------------------------
        if self.state in (FaceState.SEARCHING, FaceState.FACE_LOST):
            if self._present_streak >= self.cfg.FACE_HIT_DEBOUNCE_FRAMES:
                self._set_state(FaceState.STABILIZING, now)
            else:
                if self._absence_started_at is None:
                    self._absence_started_at = now
                if self.state == FaceState.SEARCHING and \
                        (now - self._absence_started_at) >= self.cfg.FACE_LOST_SEARCHING_SECONDS:
                    self._set_state(FaceState.FACE_LOST, now)

        elif self.state == FaceState.STABILIZING:
            if self._absent_streak >= self.cfg.FACE_MISS_DEBOUNCE_FRAMES:
                self._set_state(FaceState.SEARCHING, now)
            elif (now - self._state_entered_at) >= self.cfg.FACE_STABILITY_SECONDS:
                self._set_state(FaceState.TRACKING, now)

        elif self.state == FaceState.TRACKING:
            if self._absent_streak >= self.cfg.FACE_MISS_DEBOUNCE_FRAMES:
                self._set_state(FaceState.SEARCHING, now)
            elif self._occluded_streak >= self.cfg.OCCLUSION_CONFIRM_FRAMES:
                self._set_state(FaceState.OCCLUDED, now)

        elif self.state == FaceState.OCCLUDED:
            if self._absent_streak >= self.cfg.FACE_MISS_DEBOUNCE_FRAMES:
                self._set_state(FaceState.SEARCHING, now)
            elif self._clear_streak >= self.cfg.OCCLUSION_CLEAR_FRAMES:
                self._set_state(FaceState.TRACKING, now)

        return prev_state != self.state

    # --------------------------------------------------------------------
    def _set_state(self, new_state, now):
        self.state = new_state
        self._state_entered_at = now
        if new_state == FaceState.TRACKING:
            if self._tracking_entered_at is None:
                self._tracking_entered_at = now
        else:
            self._tracking_entered_at = None
        if new_state in (FaceState.STABILIZING, FaceState.TRACKING):
            self._absence_started_at = None

    # --------------------------------------------------------------------
    @property
    def is_tracking(self):
        return self.state == FaceState.TRACKING

    def duration_in_state(self, now):
        """How long we've continuously been in the CURRENT state."""
        return max(0.0, now - self._state_entered_at)

    def absent_duration(self, now):
        """
        Total time since the face was last confidently present. Meaningful
        during SEARCHING and FACE_LOST only; returns 0.0 otherwise (there is
        no single "no_face_start_time" anywhere else in the project any more
        - this is the only place that value is tracked).
        """
        if self.state not in (FaceState.SEARCHING, FaceState.FACE_LOST):
            return 0.0
        if self._absence_started_at is None:
            return 0.0
        return max(0.0, now - self._absence_started_at)

    def confidence(self, now, pose_jitter_confidence):
        """
        Unified confidence score used by EVERY module (head pose gating,
        dashboard readout, future alert gating). Exactly 0.0 unless
        TRACKING; ramps 0 -> 1 over the first CONFIDENCE_RAMP_FRAMES-
        equivalent seconds of a tracking session (fps-normalized against a
        30fps baseline) so a just-reacquired face cannot instantly report
        full confidence, then blends in the caller-supplied jitter-based
        pose-stability score.
        """
        if not self.is_tracking or self._tracking_entered_at is None:
            return 0.0
        ramp_seconds = self.cfg.CONFIDENCE_RAMP_FRAMES / 30.0
        ramp = clamp((now - self._tracking_entered_at) / max(ramp_seconds, 1e-6), 0.0, 1.0)
        return clamp(ramp * pose_jitter_confidence, 0.0, 1.0)


# =====================================================================================
# ALARM MANAGER (non-blocking sound)
# =====================================================================================
class AlarmManager:
    """
    Plays an audible alert while a dangerous condition is ACTIVE, using ONE
    persistent background thread for the whole ACTIVE period (never spawns a
    new thread per trigger/frame - req #7). Sound priority, per requirement:
        1. WAV file playback via winsound.PlaySound() - routes through the
           normal Windows audio mixer/output device, so it is reliably
           audible through real speakers. This is the fix for
           "winsound.Beep() is not reliably audible": Beep uses a much more
           limited synthesis path that is often very quiet or effectively
           silent on modern hardware/driver combinations.
        2. winsound.Beep() - automatic fallback if WAV playback itself fails
           for any reason (missing/corrupt file, driver issue).
        3. Terminal bell - ABSOLUTE last resort only (e.g. non-Windows with
           neither of the above available) - never the primary/main
           fallback, per requirement #2.
    Each play is SYNCHRONOUS within this dedicated thread only (not
    SND_ASYNC), so repeats can never overlap themselves - and because it is
    a background thread, it never blocks the camera loop either.
    """

    def __init__(self, wav_path):
        self.wav_path = wav_path
        self._stop_flag = threading.Event()
        self._active = False
        self._thread = None
        self._lock = threading.Lock()
        self.last_error = None

    def start(self):
        with self._lock:
            if self._active:
                return  # already running - never spins up a second thread
            self._active = True
            self._stop_flag.clear()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def stop(self):
        with self._lock:
            if not self._active:
                return
            self._active = False
            self._stop_flag.set()

    def is_active(self):
        with self._lock:
            return self._active

    def _loop(self):
        while not self._stop_flag.is_set():
            self._play_once()
            self._stop_flag.wait(Config.ALARM_REPEAT_COOLDOWN_SECONDS)

    def _play_once(self):
        # 1) WAV via winsound.PlaySound - SYNCHRONOUS (no SND_ASYNC), so it
        # fully finishes here before the next repeat and can never overlap
        # itself or race with a Beep fallback.
        if _HAS_WINSOUND and self.wav_path and os.path.isfile(self.wav_path):
            try:
                winsound.PlaySound(self.wav_path, winsound.SND_FILENAME)
                self.last_error = None
                return
            except Exception as exc:
                # FIX (req #13): recorded, not discarded - surfaced by
                # get_last_error() / diagnostics.
                self.last_error = f"WAV playback failed: {exc}"
                # fall through to Beep

        # 2) winsound.Beep fallback
        if _HAS_WINSOUND:
            try:
                winsound.Beep(Config.BEEP_FREQ, Config.BEEP_DURATION_MS)
                self.last_error = None
                return
            except Exception as exc:
                self.last_error = f"Beep failed: {exc}"

        # 3) Terminal bell - absolute last resort only, never the main path
        # (req #2). Only reached here if winsound isn't available at all.
        try:
            sys.stdout.write('\a')
            sys.stdout.flush()
        except Exception as exc:
            self.last_error = f"All audio outputs failed: {exc}"

    def self_test(self):
        """
        Runs BOTH the WAV and Beep mechanisms independently at startup and
        prints the real, honest result of each - req #6/#13/#14: never
        assumes success, never silently swallows the actual exception.
        Returns (beep_ok, wav_ok).
        """
        print("[Audio] TEST START")

        beep_ok, beep_err = self._test_beep()
        if beep_ok:
            print("[Audio] BEEP: OK")
        else:
            print(f"[Audio] BEEP: FAILED: {beep_err}")

        wav_ok, wav_err = self._test_wav()
        if wav_ok:
            print("[Audio] WAV: OK")
        else:
            print(f"[Audio] WAV: FAILED: {wav_err}")

        return beep_ok, wav_ok

    def _test_beep(self):
        if not _HAS_WINSOUND:
            return False, "winsound module not available on this OS"
        try:
            winsound.Beep(Config.BEEP_FREQ, 120)
            return True, None
        except Exception as exc:
            return False, str(exc)

    def _test_wav(self):
        if not (self.wav_path and os.path.isfile(self.wav_path)):
            return False, f"WAV file not found at {self.wav_path}"
        if not _HAS_WINSOUND:
            return False, "winsound module not available on this OS"
        try:
            winsound.PlaySound(self.wav_path, winsound.SND_FILENAME)
            return True, None
        except Exception as exc:
            return False, str(exc)


# =====================================================================================
# NEW: VOICE ALERT MANAGER (non-blocking, queued text-to-speech)
# =====================================================================================
class VoiceAlertManager:
    """
    Speaks short alert phrases using pyttsx3 on a dedicated background thread so
    it never blocks the video loop. Falls back to a no-op if pyttsx3 is not
    installed - the rest of the system keeps working with beep alarms only.
    """

    def __init__(self):
        self._queue = collections.deque()
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop_flag = threading.Event()
        self._engine = None
        self.init_error = None
        self.last_error = None

        if not _HAS_PYTTSX3:
            self.init_error = "pyttsx3 is not installed (pip install pyttsx3)"
        else:
            try:
                self._engine = pyttsx3.init()
                self._engine.setProperty('rate', 170)
            except Exception as exc:
                # FIX (req #4 - "print the real exception"): previously this
                # exception was discarded entirely, so a broken TTS backend
                # (missing OS speech driver, etc.) looked identical to
                # "pyttsx3 not installed" with zero way to diagnose it.
                self._engine = None
                self.init_error = str(exc)

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def speak(self, text):
        """Queue a phrase to be spoken (skips duplicate back-to-back phrases)."""
        if self._engine is None:
            return
        with self._lock:
            if not self._queue or self._queue[-1] != text:
                self._queue.append(text)
        self._wake.set()

    def _loop(self):
        while not self._stop_flag.is_set():
            self._wake.wait(timeout=0.5)
            self._wake.clear()
            while True:
                with self._lock:
                    if not self._queue:
                        break
                    text = self._queue.popleft()
                try:
                    self._engine.say(text)
                    self._engine.runAndWait()
                    self.last_error = None
                except Exception as exc:
                    self.last_error = str(exc)

    def is_available(self):
        return self._engine is not None

    def stop(self):
        self._stop_flag.set()
        self._wake.set()


# =====================================================================================
# FIX (CRITICAL): MEDIAPIPE COMPATIBILITY LAYER
# =====================================================================================
# ROOT CAUSE of "AttributeError: module 'mediapipe' has no attribute 'solutions'":
# newer MediaPipe pip wheels (this project's installed version included) ship
# ONLY the newer Tasks API (mediapipe.tasks.python.vision.FaceLandmarker) and
# do not expose mp.solutions AT ALL - not even attempting to touch
# mp.solutions.face_mesh is safe on those builds; simply importing/using it
# raises immediately. Older MediaPipe installs still have mp.solutions.
#
# FaceMeshCompat below detects, at runtime, which API this exact installed
# MediaPipe actually offers and uses that one - it NEVER assumes mp.solutions
# exists. Whichever backend it picks, it exposes the IDENTICAL interface the
# rest of this project already calls everywhere else:
#       self.face_mesh.process(rgb_frame) -> object with .multi_face_landmarks
#       self.face_mesh.close()
# ...where multi_face_landmarks[0].landmark[i].x/.y/.z has the exact same
# shape/semantics as the legacy API. This means EVERY downstream function in
# this file (EAR, MAR, head pose, occlusion, bbox, drawing) needed ZERO
# changes - only the initialization/acquisition layer changes.
# =====================================================================================
FACE_LANDMARKER_MODEL_FILENAME = "face_landmarker.task"
FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)


class _LandmarkListShim:
    """Wraps a flat list of Tasks-API landmarks so it exposes `.landmark`,
    exactly like the legacy API's per-face object did."""
    __slots__ = ("landmark",)

    def __init__(self, landmark_list):
        self.landmark = landmark_list


class _FaceMeshResultShim:
    """Wraps a Tasks-API FaceLandmarkerResult so it exposes
    `.multi_face_landmarks`, exactly like the legacy API's process() result."""
    __slots__ = ("multi_face_landmarks",)

    def __init__(self, multi_face_landmarks):
        self.multi_face_landmarks = multi_face_landmarks


class FaceMeshCompat:
    """
    Drop-in replacement for `mp.solutions.face_mesh.FaceMesh` that works
    whether the installed MediaPipe exposes the legacy `solutions` API or
    only the newer Tasks API. NEVER touches `mp.solutions` unless it has
    first confirmed (via hasattr) that it actually exists.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.backend = None          # "legacy" or "tasks", for diagnostics
        self.init_error = None
        self._legacy_face_mesh = None
        self._landmarker = None
        self._last_timestamp_ms = -1

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            # ---- Legacy solutions API (present on older MediaPipe installs) --
            self.backend = "legacy"
            mp_face_mesh = mp.solutions.face_mesh
            self._legacy_face_mesh = mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=cfg.MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=cfg.MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
            )
            print("[MediaPipe] Legacy 'solutions.face_mesh' API detected - using it.")
        else:
            # ---- Newer Tasks API (mp.solutions is not present at all here) ---
            self.backend = "tasks"
            model_path = self._ensure_model_downloaded()
            from mediapipe.tasks import python as mp_tasks_python
            from mediapipe.tasks.python import vision as mp_vision

            base_options = mp_tasks_python.BaseOptions(model_asset_path=model_path)
            options = mp_vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=cfg.MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
                min_face_presence_confidence=cfg.MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=cfg.MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
            print("[MediaPipe] 'solutions' API not available on this install - "
                  f"using Tasks API FaceLandmarker (model: {model_path}).")

    # ------------------------------------------------------------------
    def _ensure_model_downloaded(self):
        """
        Locates (or downloads) the .task model the Tasks API needs.
          1. Reuse a local 'face_landmarker.task' next to this script if present
             (lets users pre-place/offline-load it - no internet needed then).
          2. Otherwise auto-download it from Google's official MediaPipe model
             bucket (one-time, a few MB).
        Never silently disables face detection - raises a clear RuntimeError
        with the exact manual-fix steps if both fail.
        """
        try:
            here = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            here = os.getcwd()
        local_path = os.path.join(here, FACE_LANDMARKER_MODEL_FILENAME)

        if os.path.isfile(local_path) and os.path.getsize(local_path) > 0:
            return local_path

        print("[MediaPipe] face_landmarker.task not found locally - "
              "downloading it once (a few MB)...")
        try:
            urllib.request.urlretrieve(FACE_LANDMARKER_MODEL_URL, local_path)
        except Exception as exc:
            self.init_error = str(exc)
        if os.path.isfile(local_path) and os.path.getsize(local_path) > 0:
            print(f"[MediaPipe] Model downloaded successfully to: {local_path}")
            return local_path

        raise RuntimeError(
            "MediaPipe FaceLandmarker model could not be found or downloaded.\n"
            f"  Expected at : {local_path}\n"
            f"  Download URL: {FACE_LANDMARKER_MODEL_URL}\n"
            "  Fix: manually download that file and place it next to this "
            "script as 'face_landmarker.task', then re-run."
        )

    # ------------------------------------------------------------------
    def process(self, rgb_frame):
        """Same contract as legacy FaceMesh.process(): returns an object
        with `.multi_face_landmarks` (a list, empty/None if no face)."""
        if self.backend == "legacy":
            return self._legacy_face_mesh.process(rgb_frame)

        # ---- Tasks API path -------------------------------------------
        rgb_frame = np.ascontiguousarray(rgb_frame)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # VIDEO mode requires strictly increasing integer millisecond
        # timestamps - guard against duplicate/out-of-order timestamps
        # (can happen on some systems' clock resolution).
        timestamp_ms = int(time.time() * 1000)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.face_landmarks:
            return _FaceMeshResultShim(None)

        # Wrap each detected face's flat landmark list so downstream code's
        # `face_landmarks.landmark[i].x/.y/.z` keeps working exactly as it
        # did against the legacy API's output shape (478 landmarks incl.
        # iris, same topology/indices as refine_landmarks=True).
        wrapped = [_LandmarkListShim(face) for face in result.face_landmarks]
        return _FaceMeshResultShim(wrapped)

    # ------------------------------------------------------------------
    def close(self):
        try:
            if self.backend == "legacy" and self._legacy_face_mesh is not None:
                self._legacy_face_mesh.close()
            elif self.backend == "tasks" and self._landmarker is not None:
                self._landmarker.close()
        except Exception as exc:
            print(f"[MediaPipe] Cleanup warning: {exc}")


# =====================================================================================
# UTILITY MATH
# =====================================================================================
def euclidean_distance(p1, p2):
    # Works for 2D (x, y) and 3D (x, y, z) points. Every pre-existing caller
    # passes 2D points, for which this is exactly the original 2D formula. The
    # 3D form is what makes the mouth ratio independent of head pitch/yaw.
    n = min(len(p1), len(p2))
    return math.sqrt(sum((float(p1[i]) - float(p2[i])) ** 2 for i in range(n)))


def eye_aspect_ratio(landmarks_px, indices):
    p1 = landmarks_px[indices[0]]
    p2 = landmarks_px[indices[1]]
    p3 = landmarks_px[indices[2]]
    p4 = landmarks_px[indices[3]]
    p5 = landmarks_px[indices[4]]
    p6 = landmarks_px[indices[5]]
    vertical_1 = euclidean_distance(p2, p6)
    vertical_2 = euclidean_distance(p3, p5)
    horizontal = euclidean_distance(p1, p4)
    if horizontal == 0:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def mouth_aspect_ratio(landmarks_px, indices):
    """
    NEW: Mouth Aspect Ratio (MAR) for yawn detection.
    indices = [left_corner, right_corner, top_outer, bottom_outer, top_inner, bottom_inner]
    Mirrors the EAR formula: (vertical1 + vertical2) / (2 * horizontal).
    """
    left = landmarks_px[indices[0]]
    right = landmarks_px[indices[1]]
    top_outer = landmarks_px[indices[2]]
    bottom_outer = landmarks_px[indices[3]]
    top_inner = landmarks_px[indices[4]]
    bottom_inner = landmarks_px[indices[5]]
    horizontal = euclidean_distance(left, right)
    if horizontal == 0:
        return 0.0
    vertical_1 = euclidean_distance(top_outer, bottom_outer)
    vertical_2 = euclidean_distance(top_inner, bottom_inner)
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def rotation_matrix_to_euler_angles(rmat):
    sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        pitch = math.atan2(-rmat[2, 0], sy)
        yaw = math.atan2(rmat[1, 0], rmat[0, 0])
        roll = math.atan2(rmat[2, 1], rmat[2, 2])
    else:
        pitch = math.atan2(-rmat[2, 0], sy)
        yaw = 0.0
        roll = math.atan2(-rmat[1, 2], rmat[1, 1])
    return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))


def clamp(value, low, high):
    return max(low, min(high, value))


# =====================================================================================
# NEW: YOLO OBJECT DETECTOR (shared base for phone / seat-belt detection)
# =====================================================================================
class _YoloDetectorBase:
    """
    Thin wrapper around an Ultralytics YOLO model. Loads lazily on first use so
    the whole program still starts even if `ultralytics` isn't installed or a
    model file is missing - the relevant feature just disables itself.

    pip install ultralytics
    """

    def __init__(self, model_path, enabled=True):
        self.model_path = model_path
        self.enabled = enabled
        self.model = None
        self.load_error = None
        self._tried_load = False

    def _ensure_loaded(self):
        if self._tried_load:
            return self.model is not None
        self._tried_load = True
        if not self.enabled:
            return False
        try:
            from ultralytics import YOLO  # imported lazily (optional dependency)
            self.model = YOLO(self.model_path)
        except Exception as exc:
            self.load_error = str(exc)
            self.model = None
        return self.model is not None

    def is_available(self):
        return self._ensure_loaded()

    def detect(self, frame, conf):
        """Returns a list of (x1, y1, x2, y2, cls_id, confidence) boxes."""
        if not self._ensure_loaded():
            return []
        try:
            results = self.model.predict(frame, conf=conf, verbose=False)
            boxes = []
            for r in results:
                if r.boxes is None:
                    continue
                for b in r.boxes:
                    xyxy = b.xyxy[0].tolist()
                    cls_id = int(b.cls[0].item())
                    score = float(b.conf[0].item())
                    boxes.append((xyxy[0], xyxy[1], xyxy[2], xyxy[3], cls_id, score))
            return boxes
        except Exception:
            return []


class PhoneDetector(_YoloDetectorBase):
    """
    NEW (Feature 4): Detects mobile phones using a stock COCO-pretrained YOLO
    model (auto-downloaded by ultralytics as "yolov8n.pt" on first run - no
    manual weight placement needed). COCO class id 67 = "cell phone".
    """
    COCO_CELL_PHONE_CLASS_ID = 67

    def __init__(self, model_path, enabled=True):
        super().__init__(model_path, enabled)

    def find_phone_boxes(self, frame, conf):
        boxes = self.detect(frame, conf)
        return [b for b in boxes if b[4] == self.COCO_CELL_PHONE_CLASS_ID]


class SeatbeltDetector(_YoloDetectorBase):
    """
    NEW (Feature 5): Detects seat-belt presence. Stock COCO models have NO
    seat-belt class, so this expects a custom-trained YOLO model
    (e.g. fine-tuned yolov8n on a seat-belt dataset) at YOLO_SEATBELT_MODEL_PATH.
    Convention expected of that custom model: class 0 = "seatbelt",
    class 1 = "no_seatbelt" (adjust `SEATBELT_CLASS_ID` / `NO_SEATBELT_CLASS_ID`
    to match whatever your trained model actually uses).
    If the weight file isn't found, this feature cleanly disables itself and
    the rest of the DMS keeps running.
    """
    SEATBELT_CLASS_ID = 0
    NO_SEATBELT_CLASS_ID = 1

    def __init__(self, model_path, enabled=True):
        import os
        self._file_exists = os.path.isfile(model_path)
        super().__init__(model_path, enabled and self._file_exists)

    def classify(self, frame, conf):
        """Returns True (belt on), False (no belt), or None (no detection)."""
        boxes = self.detect(frame, conf)
        if not boxes:
            return None
        best = max(boxes, key=lambda b: b[5])
        return best[4] == self.SEATBELT_CLASS_ID


# =====================================================================================
# NEW (Feature 6): TRIP REPORT GENERATOR (PDF)
# =====================================================================================
class TripReportGenerator:
    """
    Builds a one-page PDF trip summary when the program exits.
    pip install reportlab
    """

    def __init__(self, output_path):
        self.output_path = output_path

    def generate(self, stats: dict):
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib import colors
            from reportlab.lib.units import mm
            from reportlab.pdfgen import canvas
        except ImportError:
            print("[TripReport] 'reportlab' not installed - skipping PDF. "
                  "Run: pip install reportlab")
            return None

        try:
            c = canvas.Canvas(self.output_path, pagesize=A4)
            width, height = A4

            c.setFillColor(colors.HexColor("#1a1a1a"))
            c.rect(0, height - 30 * mm, width, 30 * mm, fill=1, stroke=0)
            c.setFillColor(colors.white)
            c.setFont("Helvetica-Bold", 20)
            c.drawString(15 * mm, height - 15 * mm, "Driver Monitoring System")
            c.setFont("Helvetica", 12)
            c.drawString(15 * mm, height - 23 * mm, "Trip Safety Report")

            y = height - 42 * mm
            c.setFillColor(colors.black)

            def row(label, value, danger=False):
                nonlocal y
                c.setFont("Helvetica-Bold", 11)
                c.drawString(15 * mm, y, f"{label}:")
                c.setFont("Helvetica", 11)
                c.setFillColor(colors.red if danger else colors.black)
                c.drawString(80 * mm, y, str(value))
                c.setFillColor(colors.black)
                y -= 8 * mm

            row("Date", stats.get("date", "-"))
            row("Driving Time", stats.get("driving_time", "-"))
            row("Average FPS", stats.get("avg_fps", "-"))
            row("Blink Count", stats.get("blink_count", "-"))
            row("Yawn Count", stats.get("yawn_count", "-"),
                danger=stats.get("yawn_count", 0) and stats["yawn_count"] > 5)
            row("Fatigue Score", stats.get("fatigue_score", "-"))
            row("Phone Usage Count", stats.get("phone_usage_count", "-"),
                danger=stats.get("phone_usage_count", 0) > 0)
            row("Distraction Count", stats.get("distraction_count", "-"))
            row("Head Alerts", stats.get("head_alerts", "-"))
            row("Seat Belt Status", stats.get("seatbelt_status", "-"),
                danger=stats.get("seatbelt_status") == "NOT WORN")

            y -= 4 * mm
            c.setFont("Helvetica-Bold", 13)
            rating = stats.get("safety_rating", "-")
            c.drawString(15 * mm, y, f"Overall Safety Rating: {rating}")

            c.showPage()
            c.save()
            return self.output_path
        except Exception as exc:
            print(f"[TripReport] Failed to generate PDF: {exc}")
            return None


# =====================================================================================
# NEW (Feature 7): FIREBASE FIRESTORE UPLOAD MANAGER
# =====================================================================================
class FirebaseManager:
    """
    Uploads the trip-report dict to Firebase Firestore on a background thread
    so it never blocks shutdown.

    ---------------------------------------------------------------------
    SETUP INSTRUCTIONS
    ---------------------------------------------------------------------
    1. pip install firebase-admin
    2. Go to Firebase Console -> Project Settings -> Service Accounts
       -> "Generate new private key". This downloads a JSON credentials file.
    3. Save it next to this script as `firebase_credentials.json`
       (or update Config.FIREBASE_CREDENTIALS_JSON to point to it).
    4. Create a Firestore database in the Firebase console (Native mode).
    5. Set Config.ENABLE_FIREBASE = True and set Config.DRIVER_NAME.
    ---------------------------------------------------------------------
    """

    def __init__(self, credentials_path, collection_name, enabled=True):
        self.credentials_path = credentials_path
        self.collection_name = collection_name
        self.enabled = enabled
        self._db = None
        self._init_error = None
        if self.enabled:
            self._init()

    def _init(self):
        import os
        if not os.path.isfile(self.credentials_path):
            self._init_error = (
                f"Firebase credentials file not found at '{self.credentials_path}'. "
                f"See FirebaseManager docstring for setup steps."
            )
            self.enabled = False
            return
        try:
            import firebase_admin
            from firebase_admin import credentials, firestore
            if not firebase_admin._apps:
                cred = credentials.Certificate(self.credentials_path)
                firebase_admin.initialize_app(cred)
            self._db = firestore.client()
        except Exception as exc:
            self._init_error = str(exc)
            self.enabled = False

    def upload_async(self, trip_data: dict):
        if not self.enabled or self._db is None:
            if self._init_error:
                print(f"[Firebase] Skipped upload: {self._init_error}")
            return

        def _worker():
            try:
                self._db.collection(self.collection_name).add(trip_data)
                print("[Firebase] Trip report uploaded to Firestore.")
            except Exception as exc:
                print(f"[Firebase] Upload failed: {exc}")

        threading.Thread(target=_worker, daemon=True).start()


# =====================================================================================
# DRIVER MONITORING SYSTEM
# =====================================================================================
class DriverMonitoringSystem:

    # MediaPipe FaceMesh eye landmark indices (6-point dlib-style ordering)
    RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]
    LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]

    # NEW: Mouth landmark indices for MAR (yawn detection)
    # [left_corner, right_corner, top_outer, bottom_outer, top_inner, bottom_inner]
    MOUTH_IDX = [61, 291, 0, 17, 13, 14]

    # FIX (phone-to-ear/calling detection): MediaPipe's 468/478-point face
    # mesh has no true EXTERNAL ear landmarks (it only models the front of
    # the face) - the jaw/cheek contour points nearest each ear (standard
    # proxies used across MediaPipe-based head-pose/ear-region projects) are
    # used as an approximate ear position for proximity checks.
    LEFT_EAR_LANDMARK_IDX = 234
    RIGHT_EAR_LANDMARK_IDX = 454

    # ==========================================================================
    # FIX (head pose "at the source"): landmark set + 3D reference model for the
    # head-pose solvePnP.
    #
    # PROBLEM 1 - jaw/mouth points inside the pose solve. The old set contained
    # the CHIN (152) and BOTH MOUTH CORNERS (61, 291). Those are the points that
    # move when the mouth opens, so opening the mouth fed directly into the
    # estimated head angles (and, because mouth corners move asymmetrically,
    # into YAW - i.e. a spurious HEAD LEFT/RIGHT). On synthetic faces a jaw drop
    # biased the pose by ~3 deg of pitch and tripled yaw jitter; with the rigid
    # set below it changes the pose by 0.00 deg.
    # The new set uses ONLY points that are rigid with respect to the skull:
    # nose tip, nose bridge, forehead, the four eye corners and both temples.
    #
    # PROBLEM 2 - inconsistent 3D model. The old "generic" model was Y-up and
    # not to scale with real faces, while image Y points DOWN. solvePnP therefore
    # locked onto a mirrored (~180 deg roll) solution and the code had to remap
    # Euler slots and flip signs to compensate (which is where the confusion
    # between nod and turn came from).
    # The model below is MediaPipe's own canonical face model (the reference mesh
    # these exact landmark indices come from, in cm) converted to the camera's
    # convention: x = image right, y = image DOWN, z = away from the camera
    # (canonical (x, y, z) -> (x, -y, -z)). With it, an identity rotation is a
    # face looking straight at the camera and the angles come out directly.
    #
    # NOTE ON LEFT/RIGHT: run() mirrors the frame (cv2.flip), so the image shows
    # what the driver would see in a mirror. "LEFT" therefore means the face
    # turned toward the LEFT side of the displayed image (the driver's own left),
    # which is the same convention the existing dashboard has always used.
    # ==========================================================================
    POSE_LANDMARK_IDX = {
        "nose_tip": 1,
        "nose_bridge": 168,
        "forehead": 10,
        "right_eye_outer": 33,
        "right_eye_inner": 133,
        "left_eye_inner": 362,
        "left_eye_outer": 263,
        "right_temple": 127,
        "left_temple": 356,
    }

    # 3D reference points (cm, camera convention) in POSE_LANDMARK_IDX order.
    MODEL_POINTS_3D = np.array([
        (0.000, 1.127, -7.476),   # nose_tip (landmark 1),
        (0.000, -3.271, -5.236),   # nose_bridge (landmark 168),
        (0.000, -8.262, -4.482),   # forehead (landmark 10),
        (-4.446, -2.664, -3.173),   # right_eye_outer (landmark 33),
        (-1.856, -2.585, -3.758),   # right_eye_inner (landmark 133),
        (1.856, -2.585, -3.758),   # left_eye_inner (landmark 362),
        (4.446, -2.664, -3.173),   # left_eye_outer (landmark 263),
        (-7.743, -2.365, 2.005),   # right_temple (landmark 127),
        (7.743, -2.365, 2.005),   # left_temple (landmark 356)
    ], dtype=np.float64)

    # Mouth-opening inner-lip landmarks (upper inner, lower inner) used for the
    # extra yawn check (YAWN_INNER_MIN_RATIO). Same indices as MOUTH_IDX[4:6].
    MOUTH_INNER_IDX = (13, 14)

    def __init__(self):
        self.cfg = Config()

        # FIX (CRITICAL): never touch mp.solutions directly - some installed
        # MediaPipe builds don't have it at all (Tasks-API-only wheels), which
        # is exactly what was crashing this project with
        # "AttributeError: module 'mediapipe' has no attribute 'solutions'".
        # FaceMeshCompat auto-detects which API is actually available and
        # exposes the identical .process()/.close() interface either way -
        # see its class docstring above for the full explanation.
        try:
            self.face_mesh = FaceMeshCompat(self.cfg)
            self.mediapipe_ok = True
        except Exception as exc:
            self.mediapipe_ok = False
            print(f"[Startup] MediaPipe Face Landmarker: FAILED - {exc}")
            raise RuntimeError(
                "Could not initialize MediaPipe face landmark detection. "
                "The Driver Monitoring System cannot run without it.\n"
                f"Details: {exc}"
            ) from exc

        # FIX (audio audibility, req #4): auto-generate the alert WAV next to
        # this script if it doesn't exist yet - no manual download required.
        # Never crashes startup if this fails; AlarmManager/self_test() will
        # honestly report WAV as FAILED (with the real reason) and fall back
        # to Beep, per req #12's "don't disable audio because one path failed".
        try:
            here = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            here = os.getcwd()
        wav_path = os.path.join(here, Config.ALERT_WAV_FILENAME)
        wav_path_ok, wav_gen_error = ensure_alert_wav(wav_path)
        if wav_gen_error:
            print(f"[Audio] Could not generate alert WAV file: {wav_gen_error}")
        self.alarm = AlarmManager(wav_path_ok)
        # NEW: optional spoken-voice alert manager
        self.voice = VoiceAlertManager() if Config.ENABLE_VOICE_ALERTS else None
        self.last_voice_time = 0.0
        # NEW (Feature 8): per-message-text cooldown, so the SAME alert can't
        # repeat within ALERT_COOLDOWN_SECONDS while a different, more urgent
        # message is never blocked by it.
        self._voice_message_last_time = {}
        # FIX (req #9/#10): per-alert-type cooldown for the central
        # trigger_alert() dispatcher's "[Alert] X -> SOUND" diagnostic line
        # and sound-start bookkeeping.
        self._alert_type_last_sound_time = {}
        # NEW (req #5): throttle for the optional debug console diagnostics.
        self._last_debug_print_time = 0.0

        # Blink / EAR state
        self.ear_history = collections.deque(maxlen=5)
        self._ema_ear = None   # NEW (Problem 10): EMA layer on top of the SMA above
        self.eye_currently_closed = False
        self.eye_closed_start_time = None
        self.blink_total = 0
        self.blink_timestamps = collections.deque()

        # PERCLOS tracking: (timestamp, is_closed)
        self.closure_samples = collections.deque()

        # Head pose sustained-direction tracking
        self.direction_start_time = {"LEFT": None, "RIGHT": None, "UP": None, "DOWN": None}
        self.distraction_events = collections.deque()  # timestamps of confirmed look-away alerts

        # NEW: sudden neck-movement tracking (frame-to-frame angle deltas)
        self.prev_yaw = None
        self.prev_pitch = None
        self.prev_pose_time = None
        self.last_sudden_turn_time = 0.0

        # NEW: distracted-driver flag (frequency of look-aways)
        self.distracted_flag = False

        # FPS
        self.prev_frame_time = time.time()
        self.fps = 0.0
        self.fps_smooth = collections.deque(maxlen=20)

        # Smoothed face bounding box (for tracking visual)
        self.smoothed_bbox = None
        self._bbox_size_hist = collections.deque(maxlen=10)   # NEW (Phase 1, Fix #4): real bbox-jitter signal
        self.bbox_alpha = 0.35

        # Status / alerts
        self.active_alerts = []
        self.warning_level = "GREEN"
        self.fatigue_score = 0.0

        # ---------------- Face state (Phase 1 - unified state machine) --------------
        # SINGLE source of truth for face presence/stability/occlusion. Every
        # other module reads self.face_sm instead of keeping its own
        # face_present / face_stable / no_face_start_time / face_first_seen_time
        # (those separate variables have been removed entirely).
        self.face_sm = FaceStateMachine(self.cfg)

        self.session_start_time = time.time()

        self.cap = None

        # ---------------- NEW: driver calibration state (Feature 1/3) ----------------
        self.calibrating = True
        self.calibration_start_time = None
        self.calibration_yaw_samples = []
        self.calibration_pitch_samples = []
        self.calibration_ear_samples = []
        self.calibration_mar_samples = []          # NEW: MAR baseline samples
        self.calibration_brightness_samples = []   # NEW: brightness baseline samples
        self.baseline_yaw = 0.0
        self.baseline_pitch = 0.0
        self.baseline_ear = self.cfg.EAR_THRESHOLD  # fallback until calibrated
        self.baseline_mar = 0.35                    # NEW: fallback neutral-mouth MAR
        self.baseline_brightness = 120.0            # NEW: fallback neutral brightness
        self.dynamic_ear_threshold = self.cfg.EAR_THRESHOLD
        self.dynamic_mar_threshold = self.cfg.MAR_THRESHOLD  # NEW

        # ---------------- NEW: head-pose smoothing (Feature 1) -----------------------
        self.yaw_smooth_hist = collections.deque(maxlen=self.cfg.POSE_SMOOTHING_FRAMES)
        self.pitch_smooth_hist = collections.deque(maxlen=self.cfg.POSE_SMOOTHING_FRAMES)
        self.roll_smooth_hist = collections.deque(maxlen=self.cfg.POSE_SMOOTHING_FRAMES)
        # NEW: exponential-moving-average state, layered on top of the SMA above
        self._ema_yaw = None
        self._ema_pitch = None
        self._ema_roll = None
        self._instant_streak = 0            # NEW: consecutive-frame debounce for
                                             # the "instant" extreme-angle alert
        self._head_alert_latched = {"LEFT": False, "RIGHT": False, "UP": False, "DOWN": False}
        # FIX: bridging of momentary direction dropouts (see process_head_direction)
        self._last_direction = None
        self._direction_last_seen = 0.0
        # FIX: previous solvePnP solution (initial guess for the next frame) and 3D
        # landmarks of the current frame (pose-invariant mouth ratio)
        self._pnp_prev = None
        self._pose_error_reported = False
        self._landmarks_3d = None
        # FIX (head-pose RED bug): rolling window of recent pose_confidence
        # readings - escalation is gated on the AVERAGE of these, not a
        # single instantaneous frame (see Config.POSE_CONFIDENCE_SMOOTHING_FRAMES).
        self._pose_confidence_hist = collections.deque(
            maxlen=self.cfg.POSE_CONFIDENCE_SMOOTHING_FRAMES)

        # ---------------- NEW: yawn detection state (Feature 2) ----------------------
        self.mar_hist = collections.deque(maxlen=self.cfg.YAWN_MAR_SMOOTHING_FRAMES)
        self.inner_hist = collections.deque(maxlen=self.cfg.YAWN_MAR_SMOOTHING_FRAMES)  # FIX: inner-lip ratio
        self.mouth_open_start_time = None
        self.yawn_confirmed_for_current_event = False
        self.yawn_total = 0
        self.yawning_now = False
        self.yawn_timestamps = collections.deque()   # NEW: for fatigue-score weighting
        # FIX (yawn reliability): tracks the last time the mouth was actually
        # seen open, so a brief single-frame dip below threshold doesn't
        # immediately reset mouth_open_start_time (see Config.YAWN_DIP_GRACE_SECONDS).
        self._mouth_last_open_time = None

        # ---------------- NEW: phone detection state (Feature 4) ---------------------
        self.phone_detector = PhoneDetector(self.cfg.YOLO_PHONE_MODEL_PATH,
                                             enabled=self.cfg.ENABLE_PHONE_DETECTION)
        self.phone_near_face_start_time = None
        self.phone_alert_active = False
        self.phone_usage_count = 0
        self.phone_usage_timestamps = collections.deque()  # NEW (Phase 1, Fix #9): rolling-window safety score
        self._last_phone_boxes = []          # boxes passing the normal confidence bar
        self._last_phone_boxes_all = []      # FIX: ALL boxes incl. low-confidence (ear logic only)
        self._last_phone_boxes_time = 0.0    # FIX: when a phone was last actually detected
        self._phone_face_last_seen = 0.0     # FIX: bridges YOLO flicker for the near-face timer
        self._phone_ear_last_seen = 0.0      # FIX: bridges YOLO flicker for the near-ear timer
        # FIX (phone-to-ear/calling detection): separate timer/state from the
        # existing phone-near-face tracking above - does not alter it.
        self.phone_call_start_time = None
        self.phone_call_active = False

        # ---------------- NEW: seat-belt detection state (Feature 5) -----------------
        self.seatbelt_detector = SeatbeltDetector(self.cfg.YOLO_SEATBELT_MODEL_PATH,
                                                    enabled=self.cfg.ENABLE_SEATBELT_DETECTION)
        self.seatbelt_missing_start_time = None
        self.seatbelt_alert_active = False
        self.seatbelt_status = "UNKNOWN"

        # ---------------- NEW: shared YOLO scheduling (perf) --------------------------
        # Both phone + seat-belt detectors are polled on the SAME cadence and the
        # SAME frame counter, and both reuse their last result on skipped frames -
        # this keeps YOLO inference off the hot path for 4 out of every 5 frames.
        self._yolo_frame_counter = 0

        # ---------------- NEW: fatigue-score smoothing (Feature 7) -------------------
        self.fatigue_score_smoothed = 0.0

        # ---------------- PROBLEM 6: safety-score smoothing --------------------------
        self.safety_score_smoothed = 100.0

        # ---------------- NEW: robustness state (Feature 11) -------------------------
        self._camera_fail_streak = 0
        self.low_light = False

        # ---------------- NEW: trip report + firebase (Features 6/7) -----------------
        self.trip_report_gen = TripReportGenerator(self.cfg.TRIP_REPORT_PATH)
        self.firebase = FirebaseManager(self.cfg.FIREBASE_CREDENTIALS_JSON,
                                         self.cfg.FIREBASE_COLLECTION,
                                         enabled=self.cfg.ENABLE_FIREBASE)
        self.head_alert_total = 0
        self.head_alert_timestamps = collections.deque()  # NEW (Phase 1, Fix #9): rolling-window safety score
        self._trip_report_generated = False   # NEW: guards against double-generation

        # ---------------- NEW (Part 4): "TRIP DETAILS" button + secondary panel ----
        # self._details_button_rect holds the (x1, y1, x2, y2) of the button as
        # last drawn on the composed canvas, in that canvas's own pixel
        # coordinates - the mouse callback compares click coordinates against
        # this rect directly (OpenCV mouse callbacks always report coordinates
        # in the image's native pixel space, regardless of any WINDOW_NORMAL
        # on-screen resize), so the button stays accurate even if the person
        # resizes the window.
        self._details_button_rect = None
        self.show_details = False
        self.details_window_name = "DMS | Trip Details"
        self._details_window_open = False
        self.alert_history = collections.deque(maxlen=30)   # NEW (Part 4): (timestamp, reason) log

    # ------------------------------------------------------------------------------
    def open_camera(self):
        self.cap = cv2.VideoCapture(self.cfg.CAMERA_INDEX, cv2.CAP_DSHOW) \
            if sys.platform.startswith("win") else cv2.VideoCapture(self.cfg.CAMERA_INDEX)
        if not self.cap or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.cfg.CAMERA_INDEX)
        if not self.cap.isOpened():
            raise RuntimeError("ERROR: Unable to access webcam. Check camera index / permissions.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.FRAME_HEIGHT)
        # FIX (Phase 1, goal #11): minimize the capture's internal frame queue.
        # Without this, OpenCV/the OS driver can buffer several frames deep,
        # so under any processing slowdown (YOLO cadence, etc.) the pipeline
        # ends up reading increasingly STALE frames instead of the current
        # one - perceived as growing lag/instability even when raw FPS looks
        # fine. Not all backends honor this property, so failure is silently
        # ignored rather than raised.
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

    # ==============================================================================
    # NEW (Feature 11): reopen_camera() - handles a disconnected/unplugged camera
    # by releasing the old handle and trying to reopen it a few times. Never
    # raises; returns True/False so run() can decide whether to keep going.
    # ==============================================================================
    def reopen_camera(self):
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        for attempt in range(self.cfg.CAMERA_READ_FAIL_RECONNECT_ATTEMPTS):
            try:
                self.open_camera()
                print(f"[Camera] Reconnected successfully (attempt {attempt + 1}).")
                return True
            except Exception as exc:
                print(f"[Camera] Reconnect attempt {attempt + 1} failed: {exc}")
                time.sleep(0.5)
        return False

    # ------------------------------------------------------------------------------
    def reset_session(self):
        self.ear_history.clear()
        self._ema_ear = None
        self.eye_currently_closed = False
        self.eye_closed_start_time = None
        self.blink_total = 0
        self.blink_timestamps.clear()
        self.closure_samples.clear()
        self.direction_start_time = {"LEFT": None, "RIGHT": None, "UP": None, "DOWN": None}
        self.distraction_events.clear()
        # NEW: reset sudden-movement + distracted-driver state
        self.prev_yaw = None
        self.prev_pitch = None
        self.prev_pose_time = None
        self.last_sudden_turn_time = 0.0
        self.distracted_flag = False
        self.active_alerts = []
        self.warning_level = "GREEN"
        self.fatigue_score = 0.0
        self.session_start_time = time.time()
        self.alarm.stop()

        # NEW: reset yawn / phone / seat-belt / head-alert counters (calibration
        # and baselines are intentionally kept - no need to recalibrate on 'r')
        self.mar_hist.clear()
        self.mouth_open_start_time = None
        self.yawn_confirmed_for_current_event = False
        self.yawn_total = 0
        self.yawning_now = False
        self.yawn_timestamps.clear()
        self._mouth_last_open_time = None
        self.phone_near_face_start_time = None
        self.phone_alert_active = False
        self.phone_usage_count = 0
        self.phone_usage_timestamps.clear()   # NEW (Phase 1, Fix #9)
        self.phone_call_start_time = None
        self.phone_call_active = False
        self.seatbelt_missing_start_time = None
        self.seatbelt_alert_active = False
        self.head_alert_total = 0
        self.head_alert_timestamps.clear()    # NEW (Phase 1, Fix #9)
        self._instant_streak = 0
        self._head_alert_latched = {"LEFT": False, "RIGHT": False, "UP": False, "DOWN": False}
        self._last_direction = None            # FIX
        self._direction_last_seen = 0.0        # FIX
        self._pnp_prev = None                  # FIX
        self.inner_hist.clear()                # FIX
        self._last_phone_boxes = []            # FIX
        self._last_phone_boxes_all = []        # FIX
        self._last_phone_boxes_time = 0.0      # FIX
        self._phone_face_last_seen = 0.0       # FIX
        self._phone_ear_last_seen = 0.0        # FIX
        self._pose_confidence_hist.clear()
        self.face_sm.reset()   # NEW (Phase 1): single source of truth reset
        self._bbox_size_hist.clear()   # NEW (Phase 1, Fix #4)
        self.fatigue_score_smoothed = 0.0
        self.safety_score_smoothed = 100.0
        # NEW: reset EMA pose smoothing + per-message voice cooldown bookkeeping
        self._ema_yaw = None
        self._ema_pitch = None
        self._ema_roll = None
        self._voice_message_last_time = {}

    # ------------------------------------------------------------------------------
    def landmarks_to_px(self, face_landmarks, frame_w, frame_h):
        pts = np.empty((len(face_landmarks.landmark), 2), dtype=np.float64)
        # FIX: also keep MediaPipe's own depth (z, same scale as x) so the mouth
        # ratio can be measured in 3D and therefore does not change when the head
        # pitches/yaws. The return value is unchanged (2D px) for every existing
        # caller; the 3D copy is available as self._landmarks_3d for this frame.
        pts3 = np.empty((len(face_landmarks.landmark), 3), dtype=np.float64)
        for i, lm in enumerate(face_landmarks.landmark):
            pts[i, 0] = lm.x * frame_w
            pts[i, 1] = lm.y * frame_h
            pts3[i, 0] = pts[i, 0]
            pts3[i, 1] = pts[i, 1]
            pts3[i, 2] = float(getattr(lm, "z", 0.0)) * frame_w
        self._landmarks_3d = pts3
        return pts

    # ------------------------------------------------------------------------------
    def estimate_head_pose(self, landmarks_px, frame_w, frame_h):
        """
        FIX: rewritten at the source (see the comment above POSE_LANDMARK_IDX for
        the landmark/model changes). Returns {"roll", "pitch", "yaw"} in degrees:
            yaw   > 0  -> face turned toward the RIGHT of the displayed image
            pitch > 0  -> face tilted UP   (nose rises)
            roll  > 0  -> head tilted clockwise on screen (cosmetic only)
        The angles are read from the face's forward vector, so yaw is the turn
        about the vertical axis and pitch is the nod about the horizontal axis,
        independent of each other (the old Euler-slot remap mixed them up).
        Returns None when the solve fails or is physically implausible.
        """
        try:
            image_points = np.array(
                [landmarks_px[idx] for idx in self.POSE_LANDMARK_IDX.values()],
                dtype=np.float64)[:, :2]

            focal_length = frame_w
            center = (frame_w / 2.0, frame_h / 2.0)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float64)
            dist_coeffs = np.zeros((4, 1))

            # Initial guess: the previous frame's solution when we have one (keeps
            # the solver on the same, physically-correct solution while tracking
            # and is faster), otherwise a frontal face ~60 cm from the camera.
            default_r = np.zeros((3, 1), dtype=np.float64)
            default_t = np.array([[0.0], [0.0], [60.0]], dtype=np.float64)
            prev = getattr(self, "_pnp_prev", None)
            guess_r, guess_t = prev if prev else (default_r, default_t)
            success, rotation_vec, translation_vec = cv2.solvePnP(
                self.MODEL_POINTS_3D, image_points, camera_matrix, dist_coeffs,
                guess_r.copy(), guess_t.copy(), useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE)
            if (not success or translation_vec[2, 0] <= 0) and prev:
                # the previous solution led the solver astray - retry from scratch
                success, rotation_vec, translation_vec = cv2.solvePnP(
                    self.MODEL_POINTS_3D, image_points, camera_matrix, dist_coeffs,
                    default_r.copy(), default_t.copy(), useExtrinsicGuess=True,
                    flags=cv2.SOLVEPNP_ITERATIVE)
            if not success or translation_vec[2, 0] <= 0:
                self._pnp_prev = None
                return None

            rmat, _ = cv2.Rodrigues(rotation_vec)

            # Forward vector of the face (where the nose points), camera coords.
            # Identity rotation -> (0, 0, -1): pointing back at the camera.
            fwd = rmat @ np.array([0.0, 0.0, -1.0])
            yaw = math.degrees(math.atan2(fwd[0], -fwd[2]))
            pitch = math.degrees(math.asin(clamp(-fwd[1], -1.0, 1.0)))
            yaw *= self.cfg.HEAD_POSE_YAW_SIGN
            pitch *= self.cfg.HEAD_POSE_PITCH_SIGN

            # Roll = what is left after removing the yaw/pitch rotation.
            cy, sy = math.cos(math.radians(-yaw / self.cfg.HEAD_POSE_YAW_SIGN)), \
                math.sin(math.radians(-yaw / self.cfg.HEAD_POSE_YAW_SIGN))
            cp, sp = math.cos(math.radians(-pitch / self.cfg.HEAD_POSE_PITCH_SIGN)), \
                math.sin(math.radians(-pitch / self.cfg.HEAD_POSE_PITCH_SIGN))
            r_yaw = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            r_pitch = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
            r_roll = (r_yaw @ r_pitch).T @ rmat
            roll = math.degrees(math.atan2(r_roll[1, 0], r_roll[0, 0]))

            # Plausibility filter (unchanged idea): no genuine driving pose needs
            # more than ~75 deg on either axis relative to the camera, so a result
            # beyond that is a failed/degenerate solve - skip this frame's pose
            # instead of ever feeding it into smoothing, calibration or alerts.
            IMPLAUSIBLE_ANGLE_DEG = 75.0
            if abs(pitch) > IMPLAUSIBLE_ANGLE_DEG or abs(yaw) > IMPLAUSIBLE_ANGLE_DEG:
                self._pnp_prev = None
                return None

            self._pnp_prev = (rotation_vec, translation_vec)
            return {"roll": roll, "pitch": pitch, "yaw": yaw}
        except Exception as exc:
            # Not silently swallowed any more: report the first occurrence (every
            # frame would flood the console) and still skip just this frame.
            self._pnp_prev = None
            if not getattr(self, "_pose_error_reported", False):
                self._pose_error_reported = True
                print(f"[HeadPose] pose estimation error (further repeats suppressed): {exc}")
            return None

    # ==============================================================================
    # NEW (Feature 1): smooth_pose() - moving-average smoothing over the last
    # POSE_SMOOTHING_FRAMES frames, then converts to RELATIVE angles (deviation
    # from the driver's own calibrated baseline). This is what kills the "too
    # sensitive" false positives: raw per-frame yaw/pitch from solvePnP is noisy,
    # and everyone's "neutral" pose differs slightly depending on camera mount.
    # ==============================================================================
    def smooth_pose(self, pose):
        if pose is None:
            return None
        self.yaw_smooth_hist.append(pose["yaw"])
        self.pitch_smooth_hist.append(pose["pitch"])
        self.roll_smooth_hist.append(pose["roll"])

        sma_yaw = sum(self.yaw_smooth_hist) / len(self.yaw_smooth_hist)
        sma_pitch = sum(self.pitch_smooth_hist) / len(self.pitch_smooth_hist)
        sma_roll = sum(self.roll_smooth_hist) / len(self.roll_smooth_hist)

        # ---- NEW (Feature 9/11): exponential moving average layered on top of
        # the moving average above. The SMA already kills high-frequency noise
        # over its window; this EMA removes any remaining frame-to-frame jitter
        # so the driver can move their neck naturally without the reported
        # angle "shaking" around the true value.
        alpha = self.cfg.POSE_EMA_ALPHA
        if self._ema_yaw is None:
            self._ema_yaw, self._ema_pitch, self._ema_roll = sma_yaw, sma_pitch, sma_roll
        else:
            self._ema_yaw = alpha * sma_yaw + (1 - alpha) * self._ema_yaw
            self._ema_pitch = alpha * sma_pitch + (1 - alpha) * self._ema_pitch
            self._ema_roll = alpha * sma_roll + (1 - alpha) * self._ema_roll
        smoothed_yaw, smoothed_pitch, smoothed_roll = self._ema_yaw, self._ema_pitch, self._ema_roll

        rel_yaw = smoothed_yaw - self.baseline_yaw
        rel_pitch = smoothed_pitch - self.baseline_pitch
        # Un-deadzoned copies - purely for the dashboard readout so the driver can
        # see the true signed angle; ALL classification below uses the deadzoned
        # rel_yaw / rel_pitch exactly as before.
        rel_yaw_raw, rel_pitch_raw = rel_yaw, rel_pitch

        # Deadzone: axis-SPECIFIC, derived from each axis's own WARNING angle
        # (yaw tolerates more than pitch - see YAW_WARNING_DEG/PITCH_WARNING_DEG).
        # Ordinary driving movements (mirror checks, small neck shifts) never
        # exceed this and are ignored completely.
        yaw_deadzone = self.cfg.YAW_WARNING_DEG * self.cfg.AXIS_DEADZONE_RATIO
        pitch_deadzone = self.cfg.PITCH_WARNING_DEG * self.cfg.AXIS_DEADZONE_RATIO
        if abs(rel_yaw) < yaw_deadzone:
            rel_yaw = 0.0
        if abs(rel_pitch) < pitch_deadzone:
            rel_pitch = 0.0

        return {
            "yaw": smoothed_yaw, "pitch": smoothed_pitch, "roll": smoothed_roll,
            "rel_yaw": rel_yaw, "rel_pitch": rel_pitch,
            "rel_yaw_raw": rel_yaw_raw, "rel_pitch_raw": rel_pitch_raw,
        }

    # ==============================================================================
    # NEW (Issue 8): _pose_confidence() - lightweight "AI" confidence score (no
    # heavy ML) based on how JITTERY the recent smoothed yaw/pitch history is.
    # Stable tracking -> low jitter -> confidence near 1.0. Noisy/unstable
    # tracking (occlusion, motion blur, poor light) -> high jitter -> confidence
    # drops toward 0.0, and head-pose WARNINGS are suppressed while it's low.
    # ==============================================================================
    def _pose_confidence(self):
        """
        PROBLEM 2/4 FIX: Face Tracking Confidence - deliberately independent of
        eye state (closing your eyes must NOT produce a high confidence reading).
        Combines three tracking-quality signals:
          - pose jitter        : how stable the smoothed yaw/pitch history is
          - bbox-size jitter    : how much the tracked face box's SIZE is
            fluctuating frame-to-frame (a real tracking-stability proxy - a
            box that keeps growing/shrinking means landmarks are unstable,
            unlike the old version which was a meaningless 1.0/0.5 constant)
          - lighting quality    : low light degrades MediaPipe's own detection
            quality, so confidence should drop there too

        NOTE: the "not enough samples yet" case no longer optimistically
        returns 1.0 - that used to happen right after a face reacquisition,
        i.e. exactly when confidence should be LOWEST. It now returns a low,
        honest starting value that rises as real jitter samples accumulate.
        (The overall ramp-during-reacquisition behavior is additionally
        handled by FaceStateMachine.confidence() - see that method.)
        """
        # FIX (Phase 1, Fix #4): honest low default instead of an optimistic
        # 1.0 while jitter history is still filling up.
        if len(self.yaw_smooth_hist) < 4:
            return 0.3
        try:
            # FIX (alert latency): jitter used to be the standard deviation of the
            # raw angle history. A deliberate head turn changes the angle by tens
            # of degrees inside that window, so merely turning the head crushed
            # the confidence below MIN_POSE_CONFIDENCE and the head alert was held
            # back until the history "settled" - a hidden extra delay. Jitter is
            # now the spread of the FRAME-TO-FRAME CHANGE (velocity): a smooth
            # turn is a near-constant velocity (low spread), while tracking noise
            # makes the velocity jump around (high spread).
            yaw_hist = list(self.yaw_smooth_hist)
            pitch_hist = list(self.pitch_smooth_hist)
            yaw_jitter = statistics.pstdev([b - a for a, b in zip(yaw_hist, yaw_hist[1:])])
            pitch_jitter = statistics.pstdev([b - a for a, b in zip(pitch_hist, pitch_hist[1:])])
        except statistics.StatisticsError:
            return 0.3
        jitter = max(yaw_jitter, pitch_jitter)
        # 0 deg/frame velocity-jitter -> 1.0; 7+ deg/frame -> 0.0
        pose_stability = clamp(1.0 - (jitter / 7.0), 0.0, 1.0)

        # FIX (Phase 1, Fix #4): real bbox-stability signal. The old version
        # was `1.0 if self.smoothed_bbox is not None else 0.5` - effectively
        # always 1.0 once any bbox existed, contributing nothing. This measures
        # actual frame-to-frame jitter in the tracked box's size (width+height),
        # which does reflect real landmark/tracking quality (a box that's still
        # jumping in size means tracking hasn't settled).
        bbox_stability = 1.0
        if self.smoothed_bbox is not None:
            bw = self.smoothed_bbox[2] - self.smoothed_bbox[0]
            bh = self.smoothed_bbox[3] - self.smoothed_bbox[1]
            box_size = bw + bh
            self._bbox_size_hist.append(box_size)
            if len(self._bbox_size_hist) >= 4 and box_size > 0:
                size_jitter = statistics.pstdev(self._bbox_size_hist) / box_size
                # 0% size jitter -> 1.0; 15%+ jitter -> 0.0
                bbox_stability = clamp(1.0 - (size_jitter / 0.15), 0.0, 1.0)
        else:
            bbox_stability = 0.0
            self._bbox_size_hist.clear()

        # Detection-quality component: dim lighting reduces MediaPipe's own
        # landmark-detection reliability even if the pose itself looks stable.
        lighting_quality = 0.75 if self.low_light else 1.0

        confidence = pose_stability * 0.6 + bbox_stability * 0.2 + lighting_quality * 0.2
        return clamp(confidence, 0.0, 1.0)

    # ==============================================================================
    # NEW (Phase 1, goal #7): detect_face_occlusion() - lightweight, ML-free
    # geometric occlusion heuristic. MediaPipe FaceMesh in this pipeline exposes
    # no per-landmark visibility/confidence score, so occlusion (hand, phone,
    # mask, object over part of the face) is inferred from anatomically
    # implausible, SCALE-INVARIANT landmark geometry instead: a covered region
    # typically causes MediaPipe to still emit a mesh, but with distorted/
    # collapsed geometry in that region. All ratios are normalized against the
    # face bbox so this works at any distance from the camera.
    # ==============================================================================
    def detect_face_occlusion(self, landmarks_px, bbox):
        if bbox is None or landmarks_px is None:
            return False
        bbox_w = max(1.0, bbox[2] - bbox[0])
        bbox_h = max(1.0, bbox[3] - bbox[1])

        try:
            l_corner = np.array(landmarks_px[self.LEFT_EYE_IDX[0]], dtype=np.float64)
            r_corner = np.array(landmarks_px[self.RIGHT_EYE_IDX[0]], dtype=np.float64)
            inter_ocular_ratio = np.linalg.norm(l_corner - r_corner) / bbox_w

            m_left = np.array(landmarks_px[self.MOUTH_IDX[0]], dtype=np.float64)
            m_right = np.array(landmarks_px[self.MOUTH_IDX[1]], dtype=np.float64)
            mouth_width_ratio = np.linalg.norm(m_left - m_right) / bbox_w

            eye_mid = (l_corner + r_corner) / 2.0
            mouth_mid = (m_left + m_right) / 2.0
            eye_to_mouth_ratio = np.linalg.norm(eye_mid - mouth_mid) / bbox_h

            return bool(
                inter_ocular_ratio < self.cfg.OCCLUSION_MIN_INTER_OCULAR_RATIO or
                mouth_width_ratio < self.cfg.OCCLUSION_MIN_MOUTH_WIDTH_RATIO or
                eye_to_mouth_ratio < self.cfg.OCCLUSION_MIN_EYE_MOUTH_RATIO
            )
        except Exception:
            # Any geometry failure is itself a sign of an unreliable/likely-occluded read.
            return True

    # ==============================================================================
    # NEW (Feature 3): run_calibration_step() - called every frame for the first
    # CALIBRATION_SECONDS of the session. Collects raw yaw/pitch/EAR/MAR samples
    # with the driver asked to look straight (mouth neutral), then freezes a
    # baseline used for all relative-angle and dynamic-threshold logic for the
    # rest of the run.
    # ==============================================================================
    def run_calibration_step(self, raw_pose, avg_ear, now, mar=None, brightness=None):
        if self.calibration_start_time is None:
            self.calibration_start_time = now

        if raw_pose is not None:
            self.calibration_yaw_samples.append(raw_pose["yaw"])
            self.calibration_pitch_samples.append(raw_pose["pitch"])
        if avg_ear > 0:
            self.calibration_ear_samples.append(avg_ear)
        if mar is not None and mar > 0:
            self.calibration_mar_samples.append(mar)
        if brightness is not None:
            self.calibration_brightness_samples.append(brightness)

        elapsed = now - self.calibration_start_time
        if elapsed >= self.cfg.CALIBRATION_SECONDS:
            # FIX (calibration bug - "Baseline pitch=-110.9"): a plain mean
            # over every sample collected during the 5s window is NOT robust -
            # a handful of frames where the driver was still settling into
            # position (glancing down, adjusting the seat/mirror, a motion-
            # blurred frame) can drag a naive average far from the driver's
            # actual normal/center pose. Per the spec: use the MEDIAN (robust
            # to outliers by construction) as the anchor, then reject any
            # sample farther than CALIBRATION_OUTLIER_REJECT_DEG from it and
            # average only the remaining "valid" samples for the final
            # baseline. EAR/MAR calibration gets the same treatment (a stray
            # blink during calibration must not drag the open-eye baseline
            # down and make the drowsiness threshold too sensitive).
            def robust_baseline(samples, reject_threshold, label):
                if not samples:
                    return None, 0, 0
                arr = np.array(samples, dtype=np.float64)
                median = float(np.median(arr))
                kept = arr[np.abs(arr - median) <= reject_threshold]
                if len(kept) == 0:
                    kept = arr  # never leave a baseline fully empty
                rejected = len(arr) - len(kept)
                baseline = float(np.mean(kept))
                if rejected > 0:
                    print(f"[Calibration] {label}: rejected {rejected}/{len(arr)} outlier "
                          f"sample(s) (median={median:.3f}, using {len(kept)} valid samples).")
                return baseline, len(kept), len(arr)

            if self.calibration_yaw_samples:
                self.baseline_yaw, kept, total = robust_baseline(
                    self.calibration_yaw_samples, self.cfg.CALIBRATION_OUTLIER_REJECT_DEG, "Yaw")
                if kept < max(3, total * 0.3):
                    print("[Calibration] WARNING: most yaw samples were rejected as outliers - "
                          "please face the camera directly and press 'r' to recalibrate.")
            if self.calibration_pitch_samples:
                self.baseline_pitch, kept, total = robust_baseline(
                    self.calibration_pitch_samples, self.cfg.CALIBRATION_OUTLIER_REJECT_DEG, "Pitch")
                if kept < max(3, total * 0.3):
                    print("[Calibration] WARNING: most pitch samples were rejected as outliers - "
                          "please face the camera directly and press 'r' to recalibrate.")
            if self.calibration_ear_samples:
                self.baseline_ear, _, _ = robust_baseline(
                    self.calibration_ear_samples, self.cfg.CALIBRATION_OUTLIER_REJECT_EAR, "EAR")
                # PROBLEM 3 FIX: closed threshold = Open_EAR * 0.72 (multiplicative,
                # per-driver), rather than a fixed subtractive margin - this scales
                # correctly regardless of the individual driver's natural eye shape.
                self.dynamic_ear_threshold = max(
                    0.05, self.baseline_ear * self.cfg.EAR_CLOSED_THRESHOLD_RATIO
                )
            if self.calibration_mar_samples:  # NEW: MAR baseline (Feature 4)
                self.baseline_mar, _, _ = robust_baseline(
                    self.calibration_mar_samples, self.cfg.CALIBRATION_OUTLIER_REJECT_EAR, "MAR")
                self.dynamic_mar_threshold = self.baseline_mar + self.cfg.MAR_CALIBRATION_MARGIN
            if self.calibration_brightness_samples:  # NEW: brightness baseline
                self.baseline_brightness = sum(self.calibration_brightness_samples) / \
                    len(self.calibration_brightness_samples)

            # Sanity clamp: baseline_pitch/yaw should never itself end up
            # implausible (defense in depth on top of the per-frame rejection
            # already done in estimate_head_pose()).
            self.baseline_yaw = clamp(self.baseline_yaw, -75.0, 75.0)
            self.baseline_pitch = clamp(self.baseline_pitch, -75.0, 75.0)

            self.calibrating = False
            print(f"[Calibration] Done. Baseline yaw={self.baseline_yaw:.1f} "
                  f"pitch={self.baseline_pitch:.1f} EAR={self.baseline_ear:.3f} "
                  f"MAR={self.baseline_mar:.3f} brightness={self.baseline_brightness:.1f} "
                  f"(dynamic closed-eye threshold={self.dynamic_ear_threshold:.3f}, "
                  f"dynamic yawn-open threshold={self.dynamic_mar_threshold:.3f})")
        return elapsed

    # ------------------------------------------------------------------------------
    def process_eye_state(self, avg_ear, now):
        # MODIFIED (Feature 3): use the per-driver calibrated dynamic threshold
        # instead of a fixed one, falling back to the static config value until
        # calibration completes.
        eyes_closed_now = avg_ear < self.dynamic_ear_threshold

        # PERCLOS sample
        self.closure_samples.append((now, eyes_closed_now))
        while self.closure_samples and now - self.closure_samples[0][0] > self.cfg.PERCLOS_WINDOW_SECONDS:
            self.closure_samples.popleft()

        if eyes_closed_now:
            if self.eye_closed_start_time is None:
                self.eye_closed_start_time = now
        else:
            # FIX (Phase 1, EAR): confirm a real blink using ELAPSED TIME the
            # eye was closed, not a raw frame count - this is the actual root
            # cause fix for goal #5 (EAR calculation). A frame-count threshold
            # silently changes meaning as FPS varies (camera load, YOLO
            # cadence, etc.); measuring the closure_span directly against
            # EAR_CONSEC_MIN_SECONDS behaves identically at any FPS.
            if self.eye_currently_closed and self.eye_closed_start_time is not None:
                closure_span = now - self.eye_closed_start_time
                # PROBLEM 4 FIX: a normal blink is ~100-300ms. Only count this
                # closure as a BLINK if it stayed shut for at least
                # EAR_CONSEC_MIN_SECONDS (rules out single-frame EAR noise) and
                # no more than BLINK_MAX_DURATION_SECONDS (0.6s); anything
                # longer is drowsiness, not a blink, and must NOT inflate the
                # blink counter / blink-rate signal.
                if self.cfg.EAR_CONSEC_MIN_SECONDS <= closure_span <= self.cfg.BLINK_MAX_DURATION_SECONDS:
                    self.blink_total += 1
                    self.blink_timestamps.append(now)
            self.eye_closed_start_time = None

        self.eye_currently_closed = eyes_closed_now

        while self.blink_timestamps and now - self.blink_timestamps[0] > self.cfg.BLINK_RATE_WINDOW_SECONDS:
            self.blink_timestamps.popleft()

        closed_duration = (now - self.eye_closed_start_time) if (eyes_closed_now and self.eye_closed_start_time) else 0.0
        return eyes_closed_now, closed_duration

    # ==============================================================================
    # PROBLEM 5: classify_eye_closure_stage() - staged drowsiness escalation
    # instead of a single "eyes closed -> immediate danger" cutoff:
    #   0.0 - 1.0s -> WARNING, 1.0 - 2.5s -> FATIGUE, 2.5 - 4.0s -> DROWSY,
    #   >= 4.0s -> CRITICAL. The audible alarm is only ever triggered by
    #   CRITICAL (see update_warning_level / update_alarm_and_voice).
    # ==============================================================================
    def classify_eye_closure_stage(self, closed_duration):
        if closed_duration <= 0.0:
            return None
        if closed_duration < self.cfg.EYE_STAGE_WARNING_SECONDS:
            return "WARNING"
        if closed_duration < self.cfg.EYE_STAGE_FATIGUE_SECONDS:
            return "FATIGUE"
        if closed_duration < self.cfg.EYE_STAGE_DROWSY_SECONDS:
            return "DROWSY"
        return "CRITICAL"

    # ==============================================================================
    # MODIFIED: process_head_direction()
    #
    # Now returns a dict (instead of a tuple) bundling together:
    #   - direction              : current raw gaze direction (or None / "CENTER")
    #   - instant_alert          : direction name if an INSTANT large-angle
    #                               movement was just detected (feature 1)
    #   - sudden_turn            : True if a fast frame-to-frame angle jump was
    #                               detected (feature 3)
    #   - sustained_alerts       : list of {direction, duration, level} for any
    #                               direction sustained past YELLOW/ORANGE/RED
    #                               thresholds (features 2 & 5)
    #   - look_away_duration     : duration (s) of the current sustained look-away
    #   - look_away_level        : GREEN/YELLOW/ORANGE/RED level of that look-away
    #   - distracted              : True if >= DISTRACTION_EVENT_LIMIT look-aways
    #                               happened within DISTRACTION_WINDOW_SECONDS
    #                               (feature 4)
    # ==============================================================================
    def process_head_direction(self, raw_pose, now):
        """
        REDESIGNED (automotive-grade): relative + EMA/SMA-smoothed angle, with
        AXIS-SPECIFIC (yaw vs pitch) bands expressed as a ratio of each axis's
        own WARNING angle (Yaw 30 deg / Pitch 22 deg "recommended"; Yaw 45 deg /
        Pitch 30 deg "emergency" - see Config):
          ratio < AXIS_DEADZONE_RATIO                     -> CENTER, fully ignored
          AXIS_DEADZONE_RATIO .. AXIS_MONITOR_START_RATIO  -> hysteresis gap, ignored
          AXIS_MONITOR_START_RATIO .. 1.0 (own WARNING_DEG) -> "monitoring", no alert
          >= own WARNING_DEG for HEAD_WARNING_SUSTAIN_SECONDS+ -> WARNING (escalates)
          >= own INSTANT_DEG (few consecutive raw frames)       -> INSTANT ALERT
        Hysteresis: once a direction has warned/instant-alerted, it cannot alert
        again until the ratio falls back below AXIS_HYSTERESIS_RESET_RATIO - this
        is what stops alerts from repeatedly re-triggering while oscillating
        right around a threshold (e.g. a normal mirror check).
        """
        direction = None          # "monitoring" direction (>= REL_MONITOR_START_DEG)
        instant_alert = None
        sudden_turn = False

        pose = self.smooth_pose(raw_pose)

        # FIX (head-pose RED bug): compute + record this frame's raw
        # confidence ONCE here (rather than freshly per-direction below), and
        # gate escalation on the ROLLING AVERAGE of recent readings instead
        # of a single instantaneous value - see Config.POSE_CONFIDENCE_SMOOTHING_FRAMES
        # and the fuller explanation at that config's definition.
        raw_confidence_now = self.face_sm.confidence(now, self._pose_confidence())
        self._pose_confidence_hist.append(raw_confidence_now)
        smoothed_pose_confidence = (
            sum(self._pose_confidence_hist) / len(self._pose_confidence_hist)
        )

        if pose is not None:
            yaw = pose["rel_yaw"]
            pitch = pose["rel_pitch"]

            # ---- NEW: axis-specific ratios (fraction of each axis's own
            # MONITOR_START angle) so yaw and pitch are compared fairly even
            # though pitch's threshold is smaller than yaw's (Yaw 30/45 deg vs
            # Pitch 22/30 deg - see Config). A ratio of 1.0 means "at this
            # axis's own monitor-start angle".
            yaw_monitor_start = self.cfg.YAW_WARNING_DEG * self.cfg.AXIS_MONITOR_START_RATIO
            pitch_monitor_start = self.cfg.PITCH_WARNING_DEG * self.cfg.AXIS_MONITOR_START_RATIO
            yaw_ratio = abs(yaw) / yaw_monitor_start if yaw_monitor_start else 0.0
            pitch_ratio = abs(pitch) / pitch_monitor_start if pitch_monitor_start else 0.0
            max_ratio = max(yaw_ratio, pitch_ratio)

            if max_ratio >= 1.0:
                if yaw_ratio >= pitch_ratio:
                    direction = "LEFT" if yaw < 0 else "RIGHT"
                else:
                    direction = "UP" if pitch > 0 else "DOWN"

            # ---------------- Instant alert: extreme angle, raw + debounced ---------
            if raw_pose is not None:
                raw_rel_yaw = raw_pose["yaw"] - self.baseline_yaw
                raw_rel_pitch = raw_pose["pitch"] - self.baseline_pitch
                raw_yaw_ratio = abs(raw_rel_yaw) / self.cfg.YAW_INSTANT_DEG if self.cfg.YAW_INSTANT_DEG else 0.0
                raw_pitch_ratio = abs(raw_rel_pitch) / self.cfg.PITCH_INSTANT_DEG if self.cfg.PITCH_INSTANT_DEG else 0.0
                raw_max_ratio = max(raw_yaw_ratio, raw_pitch_ratio)

                instant_extreme = raw_max_ratio >= 1.0
                self._instant_streak = self._instant_streak + 1 if instant_extreme else 0

                if self._instant_streak >= self.cfg.INSTANT_ALERT_CONSEC_FRAMES:
                    if raw_yaw_ratio >= raw_pitch_ratio:
                        candidate = "LEFT" if raw_rel_yaw < 0 else "RIGHT"
                    else:
                        candidate = "UP" if raw_rel_pitch > 0 else "DOWN"
                    # Hysteresis: don't re-fire the same direction until the
                    # driver has actually returned toward center.
                    if not self._head_alert_latched.get(candidate, False):
                        instant_alert = candidate
                        self._head_alert_latched[candidate] = True
                        self.head_alert_total += 1
                        self.head_alert_timestamps.append(now)  # NEW (Phase 1, Fix #9): rolling-window safety score

                # ---------------- Sudden neck-movement (fast frame-to-frame jump) ---
                if self.prev_yaw is not None and self.prev_pose_time is not None:
                    dt = now - self.prev_pose_time
                    if 0 < dt <= self.cfg.SUDDEN_TURN_MAX_TIME_SEC:
                        d_yaw = abs(raw_pose["yaw"] - self.prev_yaw)
                        d_pitch = abs(raw_pose["pitch"] - self.prev_pitch)
                        if max(d_yaw, d_pitch) >= self.cfg.SUDDEN_TURN_MIN_DELTA_DEG:
                            if now - self.last_sudden_turn_time >= self.cfg.SUDDEN_TURN_COOLDOWN_SEC:
                                sudden_turn = True
                                self.last_sudden_turn_time = now

                self.prev_yaw = raw_pose["yaw"]
                self.prev_pitch = raw_pose["pitch"]
                self.prev_pose_time = now

            # ---------------- Hysteresis reset: back near center clears all latches --
            if max_ratio < self.cfg.AXIS_HYSTERESIS_RESET_RATIO:
                for k in self._head_alert_latched:
                    self._head_alert_latched[k] = False

        # FIX (noise filtering WITHOUT restarting the timer): a single failed pose
        # frame or one frame of landmark jitter just under the monitoring angle used
        # to reset the sustain timer to zero, so holding a turned head right around
        # the threshold could keep restarting the 2 s clock. A dip shorter than
        # HEAD_DIRECTION_GRACE_SECONDS now keeps the timer running. While a dip is
        # being bridged (head_dip_frame) NO new alert may start - so a driver who
        # has already turned back can never be alerted late because of this hold.
        head_dip_frame = False
        if direction is not None:
            self._last_direction = direction
            self._direction_last_seen = now
        elif (self._last_direction is not None
                and (now - self._direction_last_seen) <= self.cfg.HEAD_DIRECTION_GRACE_SECONDS
                and self.direction_start_time.get(self._last_direction) is not None):
            direction = self._last_direction
            head_dip_frame = True
        else:
            self._last_direction = None

        # ---------------- Sustained "monitoring" -> WARNING (2s+) -------------------
        sustained_alerts = []
        look_away_duration = 0.0
        look_away_level = "GREEN"
        continuous_distraction_hit = False   # PROBLEM 9: single sustained look >= 3s

        for d in ("LEFT", "RIGHT", "UP", "DOWN"):
            if direction == d:
                if self.direction_start_time[d] is None:
                    self.direction_start_time[d] = now
                duration = now - self.direction_start_time[d]
                look_away_duration = duration

                # Only angles that have actually reached the axis's own WARNING
                # angle count toward a WARNING (Yaw 30 deg / Pitch 22 deg) -
                # being in the smaller "monitoring" band alone never escalates,
                # no matter how long it lasts.
                pose_now = pose
                reached_warning_deg = (not head_dip_frame) and pose_now is not None and (
                    abs(pose_now["rel_yaw"]) >= self.cfg.YAW_WARNING_DEG or
                    abs(pose_now["rel_pitch"]) >= self.cfg.PITCH_WARNING_DEG
                )
                # Hysteresis: once this direction has alerted, it stays in alert for as
                # long as the head remains in the monitoring band. Without this, a head
                # held right AT the warning angle flickered the level (and therefore the
                # alarm) on and off every few frames. The latch is cleared only when the
                # driver returns toward centre (see "Hysteresis reset" above).
                if self._head_alert_latched.get(d, False) and pose_now is not None:
                    reached_warning_deg = True

                # ---- NEW (Feature 6: confidence scoring) --------------------------
                # If the smoothed yaw/pitch history is jittery (noisy tracking -
                # e.g. partial occlusion, motion blur, poor lighting), trust it
                # less: don't let a WARNING escalate on unstable/low-confidence data.
                # NEW (Phase 1): confidence is now a function of the SINGLE
                # face state (ramps up after reacquiring TRACKING) blended
                # with the existing jitter-based score, instead of the raw
                # jitter score alone - this is what stops head-pose alerts
                # from firing the instant a face is reacquired.
                # FIX (head-pose RED bug): uses the rolling-average confidence
                # computed once above, not a fresh instantaneous reading -
                # a single noisy frame can no longer wipe out an otherwise
                # well-established multi-second sustained head turn.
                pose_confidence = smoothed_pose_confidence

                # ---- Sustain times (FIX: alert at 2.0 s) ------------------------------
                # HEAD_WARNING_SUSTAIN_SECONDS (2.0 s) is the ALERT threshold: that is
                # when the level becomes RED (banner + alarm + voice). YELLOW/ORANGE are
                # early, visual-only stages that lead up to it. In low light every time
                # is multiplied by HEAD_LOW_LIGHT_SUSTAIN_FACTOR (1.0 by default - the old
                # hard-coded 1.4 pushed the first alert to ~3.5 s).
                sustain_scale = self.cfg.HEAD_LOW_LIGHT_SUSTAIN_FACTOR if self.low_light else 1.0
                alert_sustain = self.cfg.HEAD_WARNING_SUSTAIN_SECONDS * sustain_scale
                orange_sustain = min(self.cfg.HEAD_ORANGE_SUSTAIN_SECONDS * sustain_scale, alert_sustain)
                yellow_sustain = min(self.cfg.HEAD_YELLOW_SUSTAIN_SECONDS * sustain_scale, orange_sustain)

                if not reached_warning_deg or duration < yellow_sustain or pose_confidence < self.cfg.MIN_POSE_CONFIDENCE:
                    level = "GREEN"
                elif duration >= alert_sustain:
                    level = "RED"
                elif duration >= orange_sustain:
                    level = "ORANGE"
                else:
                    level = "YELLOW"

                look_away_level = level

                # PROBLEM 9: a single glance that itself stays past the warning
                # angle continuously for DISTRACTION_CONTINUOUS_SECONDS (3s) is
                # "DISTRACTED DRIVER" on its own - short glances (which never
                # reach reached_warning_deg for this long) never trigger it.
                if (reached_warning_deg and duration >= self.cfg.DISTRACTION_CONTINUOUS_SECONDS
                        and pose_confidence >= self.cfg.MIN_POSE_CONFIDENCE):
                    continuous_distraction_hit = True

                if level != "GREEN" and not self._head_alert_latched.get(d, False):
                    sustained_alerts.append({"direction": d, "duration": duration, "level": level})
                    self._head_alert_latched[d] = True
                    self.head_alert_total += 1
                    self.head_alert_timestamps.append(now)  # NEW (Phase 1, Fix #9): rolling-window safety score
                    if not self.distraction_events or now - self.distraction_events[-1] > 2.0:
                        self.distraction_events.append(now)
                elif level != "GREEN":
                    # Already latched/counted - still reflect the live level on
                    # the dashboard without incrementing counters again.
                    sustained_alerts.append({"direction": d, "duration": duration, "level": level})
            else:
                self.direction_start_time[d] = None

        # ---------------- Distracted-driver frequency detection ---------------------
        while self.distraction_events and now - self.distraction_events[0] > self.cfg.DISTRACTION_WINDOW_SECONDS:
            self.distraction_events.popleft()
        self.distracted_flag = (
            len(self.distraction_events) >= self.cfg.DISTRACTION_EVENT_LIMIT
            or continuous_distraction_hit
        )

        return {
            "direction": direction,
            "instant_alert": instant_alert,
            "sudden_turn": sudden_turn,
            "sustained_alerts": sustained_alerts,
            "look_away_duration": look_away_duration,
            "look_away_level": look_away_level,
            "distracted": self.distracted_flag,
            "pose": pose,
        }

    # ==============================================================================
    # NEW (Feature 2): process_yawn() - Mouth Aspect Ratio based yawn detection.
    # A yawn is confirmed ONLY when the mouth has been continuously open for at
    # least YAWN_MIN_DURATION_SECONDS (4.5s default). Talking, smiling, and
    # singing produce short/intermittent mouth-open bursts that never cross this
    # duration, so they're naturally ignored. Exactly one yawn is counted per
    # continuous open-mouth event (a flag prevents double counting while the
    # mouth remains open past the confirmation point).
    # ==============================================================================
    # ==============================================================================
    # FIX (req #3 - yawn detection reliability): the mouth-open state now
    # tolerates a brief dip below the MAR threshold (landmark jitter mid-yawn
    # is common and previously reset the whole timer back to zero) via a
    # short grace period, while a genuinely longer gap (e.g. a pause between
    # spoken words while talking) still correctly resets it - see
    # Config.YAWN_DIP_GRACE_SECONDS for the full rationale.
    # ==============================================================================
    def process_yawn(self, mar, now, inner_ratio=None):
        """
        FIX: a yawn is decided ONLY from mouth landmarks - never from head angles.
        `mar` is the pose-invariant 3D ratio (see run()). The mouth counts as
        "open" only when BOTH hold: the smoothed MAR is above the calibrated
        threshold AND the inner lips are clearly parted (inner_ratio >=
        YAWN_INNER_MIN_RATIO). The second test is what rejects a lowered head or a
        jittery lower-lip landmark: neither opens the gap between the inner lips.
        It must then stay open for YAWN_MIN_DURATION_SECONDS (2.0 s).
        """
        self.mar_hist.append(mar)
        smoothed_mar = sum(self.mar_hist) / len(self.mar_hist)

        mouth_open_raw = smoothed_mar > self.dynamic_mar_threshold
        smoothed_inner = None
        if inner_ratio is not None:
            self.inner_hist.append(inner_ratio)
            smoothed_inner = sum(self.inner_hist) / len(self.inner_hist)
            mouth_open_raw = mouth_open_raw and smoothed_inner >= self.cfg.YAWN_INNER_MIN_RATIO

        if mouth_open_raw:
            self._mouth_last_open_time = now
            if self.mouth_open_start_time is None:
                self.mouth_open_start_time = now
                self.yawn_confirmed_for_current_event = False
            mouth_open = True
        else:
            within_grace = (
                self.mouth_open_start_time is not None and
                self._mouth_last_open_time is not None and
                (now - self._mouth_last_open_time) <= self.cfg.YAWN_DIP_GRACE_SECONDS
            )
            if within_grace:
                # Brief dip - keep the ongoing yawn-open timer running.
                mouth_open = True
            else:
                self.mouth_open_start_time = None
                self.yawn_confirmed_for_current_event = False
                self._mouth_last_open_time = None
                mouth_open = False

        open_duration = (now - self.mouth_open_start_time) if self.mouth_open_start_time is not None else 0.0

        yawn_just_confirmed = False
        if (mouth_open and open_duration >= self.cfg.YAWN_MIN_DURATION_SECONDS
                and not self.yawn_confirmed_for_current_event):
            self.yawn_total += 1
            self.yawn_confirmed_for_current_event = True
            yawn_just_confirmed = True
            self.yawn_timestamps.append(now)   # NEW: fatigue-score frequency weighting

        self.yawning_now = mouth_open and self.yawn_confirmed_for_current_event

        # NEW (req #5 debug diagnostics): human-readable yawn status.
        if self.yawning_now:
            yawn_status = "CONFIRMED"
        elif mouth_open:
            yawn_status = "POSSIBLE"
        else:
            yawn_status = "NORMAL"

        return {
            "mar": smoothed_mar,
            "inner_ratio": smoothed_inner if smoothed_inner is not None else 0.0,
            "mouth_open": mouth_open,
            "open_duration": open_duration,
            "yawning": self.yawning_now,
            "yawn_just_confirmed": yawn_just_confirmed,
            "yawn_total": self.yawn_total,
            "status": yawn_status,
        }

    # ------------------------------------------------------------------------------
    # MODIFIED: compute_fatigue_score() - adds a fatigue bonus when the
    # distracted-driver condition (feature 4) is active.
    # ------------------------------------------------------------------------------
    def compute_fatigue_score(self, now, head_info=None, yawn_info=None, phone_info=None):
        """
        REWRITTEN (Issue 5): realistic weighted fatigue scoring.
        Components (each 0-100), combined with fixed weights, then smoothed with
        an EMA for a gradual "recovery" instead of snapping around:
            Eye Closure (35%)   - PERCLOS blended with the CURRENT closure duration
            Head Distraction (20%) - current sustained look-away level/frequency
            Phone Usage (20%)      - live phone-near-face alert / recent detection
            Yawn Frequency (15%)   - confirmed yawns in the last 5 minutes (decaying)
            Blink Rate (10%)       - deviation from a normal blink rate
        """
        # ---- Eye Closure component --------------------------------------------
        if self.closure_samples:
            closed_count = sum(1 for _, c in self.closure_samples if c)
            perclos = (closed_count / len(self.closure_samples)) * 100.0
        else:
            perclos = 0.0
        closed_dur = (now - self.eye_closed_start_time) if (self.eye_currently_closed and self.eye_closed_start_time) else 0.0
        closure_live_score = clamp((closed_dur / self.cfg.EYE_CLOSED_ALARM_SECONDS) * 100.0, 0.0, 100.0)
        eye_closure_score = clamp(perclos * 0.55 + closure_live_score * 0.45, 0.0, 100.0)

        # ---- Blink-rate deviation component ------------------------------------
        elapsed_minutes = max((now - self.session_start_time) / 60.0, 1.0 / 60.0)
        window_minutes = min(elapsed_minutes, self.cfg.BLINK_RATE_WINDOW_SECONDS / 60.0)
        blink_rate = len(self.blink_timestamps) / max(window_minutes, 1e-3)
        blink_dev = abs(blink_rate - self.cfg.NORMAL_BLINKS_PER_MIN)
        blink_score = clamp((blink_dev / self.cfg.NORMAL_BLINKS_PER_MIN) * 100.0, 0.0, 100.0)

        # ---- Yawn-frequency component (decaying 5-minute window) --------------
        while self.yawn_timestamps and now - self.yawn_timestamps[0] > 300.0:
            self.yawn_timestamps.popleft()
        yawn_score = clamp(len(self.yawn_timestamps) * 25.0, 0.0, 100.0)

        # ---- Phone-usage component ----------------------------------------------
        phone_score = 0.0
        if phone_info is not None:
            if phone_info.get("phone_alert"):
                phone_score = 100.0
            elif phone_info.get("phone_detected"):
                phone_score = 35.0

        # ---- Head-distraction component ------------------------------------------
        head_score = 0.0
        if head_info is not None:
            level_map = {"GREEN": 0.0, "YELLOW": 40.0, "ORANGE": 70.0, "RED": 100.0}
            head_score = level_map.get(head_info.get("look_away_level", "GREEN"), 0.0)
            if head_info.get("distracted"):
                head_score = max(head_score, 85.0)

        raw_score = clamp(
            eye_closure_score * 0.35 +
            head_score * 0.20 +
            phone_score * 0.20 +
            yawn_score * 0.15 +
            blink_score * 0.10,
            0.0, 100.0
        )

        # ---- Smooth "recovery" via EMA (never snaps/jumps) -----------------------
        alpha = self.cfg.FATIGUE_SMOOTHING_ALPHA
        self.fatigue_score_smoothed = (alpha * raw_score) + ((1 - alpha) * self.fatigue_score_smoothed)

        return clamp(self.fatigue_score_smoothed, 0.0, 100.0), perclos, blink_rate

    # ==============================================================================
    # NEW (Issue 8 - dashboard): compute_live_safety_score() - a running 0-100
    # "Safety Score" for the dashboard (separate from the fatigue score): starts
    # at 100 and is reduced by accumulated events so far this session. Simple,
    # transparent, and monotonic within a session (only reset by 'r').
    # ==============================================================================
    def compute_live_safety_score(self, now):
        """
        FIX (Phase 1, goal #9): root cause was that yawn_total, phone_usage_count,
        and head_alert_total were lifetime-SESSION-cumulative counters that never
        decayed, so once a driver racked up a few incidents early on, the score
        could never recover even after a long stretch of subsequently perfect
        driving - it did not reflect CURRENT behavior at all. Every component
        below now counts only events within SAFETY_SCORE_WINDOW_SECONDS (a
        rolling window), like distraction_events already did, so sustained good
        driving lets the score climb back up again. Lifetime totals
        (self.yawn_total / self.phone_usage_count / self.head_alert_total) are
        left untouched and still used for the end-of-trip report/dashboard
        stats where a lifetime count is exactly what's wanted.
        """
        window = self.cfg.SAFETY_SCORE_WINDOW_SECONDS
        while self.yawn_timestamps and now - self.yawn_timestamps[0] > window:
            self.yawn_timestamps.popleft()
        while self.phone_usage_timestamps and now - self.phone_usage_timestamps[0] > window:
            self.phone_usage_timestamps.popleft()
        while self.head_alert_timestamps and now - self.head_alert_timestamps[0] > window:
            self.head_alert_timestamps.popleft()

        raw_score = 100.0
        raw_score -= min(len(self.yawn_timestamps) * 4, 20)
        raw_score -= min(len(self.phone_usage_timestamps) * 15, 40)
        raw_score -= min(len(self.head_alert_timestamps) * 2, 20)
        raw_score -= min(len(self.distraction_events) * 3, 15)
        if self.seatbelt_status == "NOT WORN":
            raw_score -= 20
        raw_score = clamp(raw_score, 0.0, 100.0)

        # PROBLEM 6: never let the DISPLAYED score jump instantly (e.g. the -20
        # seat-belt deduction above landing in a single frame) - blend toward the
        # raw target with an EMA, and additionally hard-cap the per-frame step.
        alpha = self.cfg.SAFETY_SCORE_SMOOTHING_ALPHA
        target = (alpha * raw_score) + ((1 - alpha) * self.safety_score_smoothed)
        step = clamp(
            target - self.safety_score_smoothed,
            -self.cfg.SAFETY_SCORE_MAX_STEP_PER_FRAME,
            self.cfg.SAFETY_SCORE_MAX_STEP_PER_FRAME,
        )
        self.safety_score_smoothed = clamp(self.safety_score_smoothed + step, 0.0, 100.0)
        return self.safety_score_smoothed

    # ==============================================================================
    # MODIFIED: update_warning_level()
    # Folds in instant alerts, sudden turns, distracted-driver flag, the
    # multi-level sustained head-pose result, and the fused multi-signal
    # drowsiness warning (Issue 8: decision fusion instead of one threshold).
    # NOTE: face-loss/"not stable" handling now lives entirely in run() - this
    # is only ever called while the face IS stable, so no_face_duration logic
    # was removed as dead code.
    # ==============================================================================
    def update_warning_level(self, eyes_closed_alarm, head_info, fatigue_score,
                              phone_alert=False, seatbelt_alert=False,
                              fused_drowsy_warning=False, eye_stage=None,
                              phone_calling=False, yawning_active=False):
        # Highest-priority / instantaneous triggers -> RED immediately.
        # NOTE (Problem 5): eyes_closed_alarm is only ever True at the CRITICAL
        # eye-closure stage (>= 4.0s) - see classify_eye_closure_stage() - so
        # RED (and therefore the audible alarm) never fires on eye closure
        # alone until that stage is reached.
        if (
            eyes_closed_alarm
            or head_info["instant_alert"] is not None
            or head_info["sudden_turn"]
            or head_info["distracted"]
            or head_info["look_away_level"] == "RED"
            or fatigue_score >= 70
            or phone_alert
            or phone_calling            # FIX (req #2): phone-to-ear -> RED, same as phone_alert
            or seatbelt_alert
            or yawning_active           # FIX (req #3/#4): confirmed yawn (while mouth stays
                                         # open) -> RED + sound, same alert-manager path as the
                                         # other two conditions, cleared the moment the mouth
                                         # closes (yawning_active naturally goes False then).
        ):
            return "RED"

        if head_info["look_away_level"] == "ORANGE" or fused_drowsy_warning or eye_stage == "DROWSY":
            return "ORANGE"

        if head_info["look_away_level"] == "YELLOW" or fatigue_score >= 40 or eye_stage == "FATIGUE":
            return "YELLOW"

        return "GREEN"


    # ------------------------------------------------------------------------------
    def update_face_bbox(self, landmarks_px):
        x_min = float(np.min(landmarks_px[:, 0]))
        x_max = float(np.max(landmarks_px[:, 0]))
        y_min = float(np.min(landmarks_px[:, 1]))
        y_max = float(np.max(landmarks_px[:, 1]))
        pad_x = (x_max - x_min) * 0.08
        pad_y = (y_max - y_min) * 0.12
        bbox = (x_min - pad_x, y_min - pad_y, x_max + pad_x, y_max + pad_y)

        if self.smoothed_bbox is None:
            self.smoothed_bbox = bbox
        else:
            a = self.bbox_alpha
            self.smoothed_bbox = tuple(
                self.smoothed_bbox[i] * (1 - a) + bbox[i] * a for i in range(4)
            )
        return self.smoothed_bbox

    # ==============================================================================
    # NEW (Feature 11): check_and_enhance_low_light() - measures mean grayscale
    # brightness; if it's below LOW_LIGHT_BRIGHTNESS_THRESHOLD, applies CLAHE
    # (localized contrast enhancement) on the L channel so FaceMesh/EAR/MAR/pose
    # stay reliable in dim cabin lighting. Never raises - on any error it just
    # returns the original frame untouched.
    # ==============================================================================
    def check_and_enhance_low_light(self, frame):
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_brightness = float(gray.mean())
            self.low_light = mean_brightness < self.cfg.LOW_LIGHT_BRIGHTNESS_THRESHOLD

            if not self.low_light or not self.cfg.ENABLE_LOW_LIGHT_ENHANCEMENT:
                return frame, mean_brightness

            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l_channel)
            enhanced_lab = cv2.merge((l_enhanced, a_channel, b_channel))
            return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR), mean_brightness
        except Exception:
            return frame, 120.0

    # ==============================================================================
    # NEW (Feature 11/perf): _prepare_yolo_frame() - downscales the frame before
    # YOLO inference (CPU cost roughly scales with pixel count) and returns the
    # scale factor so any resulting boxes can be mapped back to full-frame
    # coordinates.
    # ==============================================================================
    def _prepare_yolo_frame(self, frame):
        h, w = frame.shape[:2]
        max_side = max(h, w)
        if max_side <= self.cfg.YOLO_INFER_MAX_SIDE:
            return frame, 1.0
        scale = self.cfg.YOLO_INFER_MAX_SIDE / float(max_side)
        small = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))),
                            interpolation=cv2.INTER_LINEAR)
        return small, scale

    @staticmethod
    def _rescale_boxes(boxes, scale):
        if scale == 1.0:
            return boxes
        inv = 1.0 / scale
        return [(x1 * inv, y1 * inv, x2 * inv, y2 * inv, cls_id, conf)
                for (x1, y1, x2, y2, cls_id, conf) in boxes]

    # ==============================================================================
    # NEW (Feature 4): process_phone_detection() - runs YOLO every
    # YOLO_DETECT_EVERY_N_FRAMES frames (perf, shared cadence with seat-belt
    # detection via self._yolo_frame_counter) and reuses the previous detection
    # on skipped frames. Checks whether any detected phone box is "near" the
    # driver's face box (by center-distance AND a minimum box size, so a
    # passenger's phone or a phone visible in the background doesn't count),
    # and requires that proximity to hold for PHONE_ALERT_SECONDS before
    # raising a PHONE DISTRACTION alert.
    # ==============================================================================
    def process_phone_detection(self, frame, bbox, now, run_yolo_this_frame):
        if not self.cfg.ENABLE_PHONE_DETECTION or not self.phone_detector.is_available():
            return {"phone_detected": False, "phone_alert": False, "boxes": [], "phone_visible": False,
                    "boxes_all": []}

        # NEW (Issue 7): in low light, require a higher confidence before trusting
        # a phone detection (detections are less reliable when dim).
        strict_conf = self.cfg.PHONE_CONF_THRESHOLD + (0.15 if self.low_light else 0.0)
        ear_conf = self.cfg.PHONE_EAR_CONF_THRESHOLD + (0.10 if self.low_light else 0.0)

        if run_yolo_this_frame:
            small, scale = self._prepare_yolo_frame(frame)
            # FIX: ONE inference, at the lower of the two confidence bars. A phone
            # held to the ear is partly hidden by the hand/head and scores lower than
            # one held in the open. Boxes below the normal bar are only ever used by
            # process_phone_call_detection(), and only if they pass its ear geometry.
            boxes = self.phone_detector.find_phone_boxes(small, min(strict_conf, ear_conf))
            boxes = self._rescale_boxes(boxes, scale)
            if boxes:
                self._last_phone_boxes_all = boxes
                self._last_phone_boxes_time = now
        # Re-use the last detections only for PHONE_LOST_GRACE_SECONDS after the phone
        # was last actually seen (bridges single-frame YOLO misses, never lingers).
        if self._last_phone_boxes_all and \
                (now - self._last_phone_boxes_time) > self.cfg.PHONE_LOST_GRACE_SECONDS:
            self._last_phone_boxes_all = []
        boxes_all = self._last_phone_boxes_all
        # Normal-confidence subset: exactly what the pre-existing near-face logic used.
        boxes = [b for b in boxes_all if b[5] >= strict_conf]
        self._last_phone_boxes = boxes

        frame_area = frame.shape[0] * frame.shape[1]
        min_area = frame_area * self.cfg.PHONE_MIN_BOX_AREA_FRACTION

        phone_near_face = False
        if boxes and bbox is not None:
            fx1, fy1, fx2, fy2 = bbox
            face_w, face_h = (fx2 - fx1), (fy2 - fy1)
            # FIX: measured as the GAP between the phone box and the face box (0 when
            # they overlap), allowed up to PHONE_FACE_PROXIMITY_SCALE x the face size.
            # The old test used the box CENTRE against 1.8 x the face size from the face
            # centre - a circle that covered most of the frame, so a phone lying on a
            # passenger seat or a dashboard mount counted as "the driver's phone".
            margin = max(face_w, face_h) * self.cfg.PHONE_FACE_PROXIMITY_SCALE
            for (px1, py1, px2, py2, _cls, _conf) in boxes:
                box_area = max(0.0, px2 - px1) * max(0.0, py2 - py1)
                if box_area < min_area:
                    continue  # NEW: too small -> likely far away / background phone
                gap_x = max(fx1 - px2, 0.0, px1 - fx2)
                gap_y = max(fy1 - py2, 0.0, py1 - fy2)
                if math.hypot(gap_x, gap_y) <= margin:
                    phone_near_face = True
                    break

        # FIX: bridge short detection gaps so the 2.0 s timer is not restarted by a
        # single missed YOLO frame (it still resets once the phone is really gone).
        if phone_near_face:
            self._phone_face_last_seen = now
        elif (self.phone_near_face_start_time is not None
                and (now - self._phone_face_last_seen) <= self.cfg.PHONE_LOST_GRACE_SECONDS):
            phone_near_face = True

        phone_alert = False
        if phone_near_face:
            if self.phone_near_face_start_time is None:
                self.phone_near_face_start_time = now
            duration = now - self.phone_near_face_start_time
            if duration >= self.cfg.PHONE_ALERT_SECONDS:
                phone_alert = True
                if not self.phone_alert_active:
                    self.phone_usage_count += 1
                    self.phone_usage_timestamps.append(now)  # NEW (Phase 1, Fix #9): rolling-window safety score
                self.phone_alert_active = True
        else:
            self.phone_near_face_start_time = None
            self.phone_alert_active = False

        # ---- FIX (Part B, phone status labeling): distinguish "a phone is
        # somewhere in the visible scene" (phone_visible) from "a phone is near
        # the driver's own face" (phone_near_face/phone_detected) - a passenger's
        # phone or one visible in the background is NOT the same condition and
        # should read differently on the dashboard (see PHONE status logic in
        # _draw_calm_dashboard / draw_details_panel).
        phone_visible = bool(boxes_all)
        return {"phone_detected": phone_near_face, "phone_alert": phone_alert,
                "boxes": boxes, "phone_visible": phone_visible, "boxes_all": boxes_all}

    # ==============================================================================
    # FIX (req #2): process_phone_call_detection() - PHONE-TO-EAR / CALLING layer.
    # Separate from process_phone_detection() above (which is untouched) - reuses
    # its already-computed YOLO phone boxes (no duplicate inference) and combines
    # them with face landmarks to distinguish "phone visible near face" (e.g. held
    # up in front while texting/photographing) from "phone actually held to an
    # ear" (a call). Falls back safely to face-bbox-edge proximity if landmarks
    # are unavailable, exactly as required, rather than crashing/being skipped.
    # ==============================================================================
    def process_phone_call_detection(self, boxes, landmarks_px, bbox, frame_area, now):
        """
        `boxes` are the YOLO phone boxes already computed this frame by
        process_phone_detection() (no second inference) - run() passes the
        low-confidence-inclusive list. A box counts as "phone at the ear" only if:
          1. its centre lies on the OUTER side of that ear (lateral position
             >= PHONE_EAR_LATERAL_MIN of the half face-width, measured along the
             ear-to-ear axis) - a phone held in front of the face sits near the
             middle, so it is NOT a call; and
          2. the nearest EDGE of the box is within PHONE_EAR_PROXIMITY_SCALE x face
             width of that ear's landmark; and
          3. it passes the usual minimum-size filter, and the face itself is tracked
             (a phone somewhere in the background is never "the driver's phone").
        Left and right ears are treated symmetrically. The timer survives short
        detection gaps (PHONE_LOST_GRACE_SECONDS) and PHONE_CALL_ALERT_SECONDS (2.0 s)
        of continuous presence confirms PHONE CALLING.
        """
        phone_near_ear = False
        if boxes and bbox is not None:
            min_area = frame_area * self.cfg.PHONE_MIN_BOX_AREA_FRACTION
            fx1, fy1, fx2, fy2 = bbox
            face_h = max(1.0, fy2 - fy1)

            if landmarks_px is not None:
                # Real ear-region landmarks are available - use them (see
                # LEFT_EAR_LANDMARK_IDX/RIGHT_EAR_LANDMARK_IDX class docstring for
                # why these are the closest available proxy in this face-mesh
                # topology, which has no true external-ear points).
                try:
                    left_ear_pt = landmarks_px[self.LEFT_EAR_LANDMARK_IDX]
                    right_ear_pt = landmarks_px[self.RIGHT_EAR_LANDMARK_IDX]
                except (IndexError, TypeError):
                    landmarks_px = None  # fall through to the safe fallback below

            if landmarks_px is None:
                # FIX (req: "fall back safely... rather than crashing"): no usable
                # landmarks this frame - approximate each ear as the vertical
                # midpoint of the LEFT/RIGHT edge of the face bbox instead.
                ear_y = (fy1 + fy2) / 2.0
                left_ear_pt = (fx1, ear_y)
                right_ear_pt = (fx2, ear_y)

            lx, ly = float(left_ear_pt[0]), float(left_ear_pt[1])
            rx, ry = float(right_ear_pt[0]), float(right_ear_pt[1])
            ear_dist = math.hypot(rx - lx, ry - ly)
            # Face width reference. Floored by the face height so it does not collapse
            # when the head is turned (ear-to-ear distance shrinks with yaw).
            ref_w = max(ear_dist, 0.62 * face_h)
            ux, uy = ((rx - lx) / ear_dist, (ry - ly) / ear_dist) if ear_dist > 1e-6 else (1.0, 0.0)
            mid_x, mid_y = (lx + rx) / 2.0, (ly + ry) / 2.0
            half_w = ref_w / 2.0
            tol = self.cfg.PHONE_EAR_PROXIMITY_SCALE * ref_w
            ear_conf = self.cfg.PHONE_EAR_CONF_THRESHOLD + (0.10 if self.low_light else 0.0)

            for (px1, py1, px2, py2, _cls, conf) in boxes:
                if conf < ear_conf:
                    continue
                if max(0.0, px2 - px1) * max(0.0, py2 - py1) < min_area:
                    continue  # too small -> likely far away / background phone
                pcx, pcy = (px1 + px2) / 2.0, (py1 + py2) / 2.0
                lateral = ((pcx - mid_x) * ux + (pcy - mid_y) * uy) / half_w  # -1 = left ear, +1 = right ear
                for ex, ey, side in ((lx, ly, -1.0), (rx, ry, 1.0)):
                    if side * lateral < self.cfg.PHONE_EAR_LATERAL_MIN:
                        continue  # centre is not on this ear's side of the face
                    dx = max(px1 - ex, 0.0, ex - px2)
                    dy = max(py1 - ey, 0.0, ey - py2)
                    if math.hypot(dx, dy) <= tol:
                        phone_near_ear = True
                        break
                if phone_near_ear:
                    break

        # Separate temporal-confirmation timer from the general phone-near-face
        # one - a phone held to the ear for PHONE_CALL_ALERT_SECONDS confirms
        # PHONE CALLING; moving it away clears the alert (after the short grace).
        if phone_near_ear:
            self._phone_ear_last_seen = now
        elif (self.phone_call_start_time is not None
                and (now - self._phone_ear_last_seen) <= self.cfg.PHONE_LOST_GRACE_SECONDS):
            phone_near_ear = True   # bridge a single missed detection

        phone_calling = False
        if phone_near_ear:
            if self.phone_call_start_time is None:
                self.phone_call_start_time = now
            duration = now - self.phone_call_start_time
            if duration >= self.cfg.PHONE_CALL_ALERT_SECONDS:
                phone_calling = True
                self.phone_call_active = True
        else:
            self.phone_call_start_time = None
            self.phone_call_active = False

        return {"phone_near_ear": phone_near_ear, "phone_calling": phone_calling}

    # ==============================================================================
    # NEW (Feature 5): process_seatbelt_detection() - runs a (custom-trained)
    # YOLO model on the SAME shared cadence as phone detection and reuses the
    # previous classification on skipped frames. If no model file is available,
    # this returns "UNAVAILABLE" / no-alert and never blocks anything else.
    # ==============================================================================
    def process_seatbelt_detection(self, frame, now, run_yolo_this_frame):
        if not self.cfg.ENABLE_SEATBELT_DETECTION or not self.seatbelt_detector.is_available():
            return {"status": "UNAVAILABLE", "alert": False}

        if run_yolo_this_frame:
            small, _scale = self._prepare_yolo_frame(frame)
            effective_conf = self.cfg.SEATBELT_CONF_THRESHOLD + (0.15 if self.low_light else 0.0)
            belt_on = self.seatbelt_detector.classify(small, effective_conf)
            if belt_on is not None:
                self.seatbelt_status = "WORN" if belt_on else "NOT WORN"
            # else: keep last known status (no detection this cycle)

        alert = False
        if self.seatbelt_status == "NOT WORN":
            if self.seatbelt_missing_start_time is None:
                self.seatbelt_missing_start_time = now
            duration = now - self.seatbelt_missing_start_time
            alert = duration >= self.cfg.SEATBELT_MISSING_ALERT_SECONDS
            self.seatbelt_alert_active = alert
        else:
            self.seatbelt_missing_start_time = None
            self.seatbelt_alert_active = False

        return {"status": self.seatbelt_status, "alert": self.seatbelt_alert_active}

    # ------------------------------------------------------------------------------
    # MODIFIED: level_color() - adds the new ORANGE level.
    # ------------------------------------------------------------------------------
    def level_color(self, level):
        return {
            "GREEN": (60, 200, 60),
            "YELLOW": (0, 210, 255),
            "ORANGE": (0, 140, 255),
            "RED": (0, 0, 255),
        }.get(level, (200, 200, 200))

    # ------------------------------------------------------------------------------
    def draw_face_box(self, frame, bbox, level):
        self._draw_calm_face_box(frame, bbox, level)
        return

    _UI_THEME = {
        "bg_top": (24, 29, 36), "bg_bottom": (12, 16, 22),
        "card": (31, 38, 47), "card_border": (61, 72, 84),
        "text": (235, 241, 245), "muted": (157, 171, 181),
        "faint": (112, 126, 137), "accent": (91, 206, 190),
        "green": (104, 211, 151), "yellow": (244, 196, 92),
        "orange": (244, 146, 84), "red": (238, 103, 104),
    }

    def _ui_fonts(self):
        if hasattr(self, "_calm_fonts"):
            return self._calm_fonts
        # FIX (readability): sizes raised (10-13 px -> 13-24 px) and more font
        # locations tried than just Windows Segoe UI, so the dashboard no longer
        # silently falls back to Pillow's tiny default bitmap font elsewhere.
        sizes = {"title": 24, "subtitle": 15, "section": 14, "body": 14,
                 "small": 13, "strong": 16, "big": 19}
        candidates = [
            ("C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/segoeuib.ttf"),
            ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
            ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
            ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
            ("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
             "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
        ]
        self._calm_fonts = None
        for regular, bold in candidates:
            try:
                self._calm_fonts = {
                    "title": ImageFont.truetype(bold, sizes["title"]),
                    "subtitle": ImageFont.truetype(regular, sizes["subtitle"]),
                    "section": ImageFont.truetype(bold, sizes["section"]),
                    "body": ImageFont.truetype(regular, sizes["body"]),
                    "small": ImageFont.truetype(regular, sizes["small"]),
                    "strong": ImageFont.truetype(bold, sizes["strong"]),
                    "big": ImageFont.truetype(bold, sizes["big"]),
                }
                break
            except Exception:
                continue
        if self._calm_fonts is None:
            try:
                default = ImageFont.load_default(size=14)   # Pillow >= 10.1: scalable
            except TypeError:
                default = ImageFont.load_default()
            self._calm_fonts = {key: default for key in
                                ("title", "subtitle", "section", "body", "small", "strong", "big")}
        return self._calm_fonts

    def _ui_level_color(self, level):
        return self._UI_THEME.get({"GREEN": "green", "YELLOW": "yellow",
                                   "ORANGE": "orange", "RED": "red"}.get(level), "muted")

    @staticmethod
    def _ui_gradient(size, top, bottom):
        width, height = size
        image = Image.new("RGB", (1, height))
        pixels = image.load()
        for row in range(height):
            ratio = row / max(1, height - 1)
            pixels[0, row] = tuple(int(top[i] + (bottom[i] - top[i]) * ratio) for i in range(3))
        return image.resize((width, height)).convert("RGBA")

    @staticmethod
    def _ui_blit(frame, image, x, y):
        height, width = frame.shape[:2]
        right, bottom = min(width, x + image.width), min(height, y + image.height)
        if right <= x or bottom <= y:
            return
        rgba = np.asarray(image.crop((0, 0, right - x, bottom - y)))
        rgb = rgba[:, :, :3][:, :, ::-1].astype(np.float32)
        alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
        original = frame[y:bottom, x:right].astype(np.float32)
        frame[y:bottom, x:right] = (rgb * alpha + original * (1 - alpha)).astype(np.uint8)

    def _draw_calm_face_box(self, frame, bbox, level):
        if bbox is None:
            return
        x1, y1, x2, y2 = [int(value) for value in bbox]
        red, green, blue = self._ui_level_color(level)
        color = (blue, green, red)
        length = max(int(min(x2 - x1, y2 - y1) * 0.16), 14)
        thickness = 2 if level == "GREEN" else 4     # warnings are drawn heavier
        for x, y, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1),
                             (x1, y2, 1, -1), (x2, y2, -1, -1)):
            cv2.line(frame, (x, y), (x + dx * length, y), color, thickness, cv2.LINE_AA)
            cv2.line(frame, (x, y), (x, y + dy * length), color, thickness, cv2.LINE_AA)

    def _draw_calm_calibration(self, frame, elapsed):
        fonts = self._ui_fonts()
        theme = self._UI_THEME
        height, width = frame.shape[:2]
        progress = clamp(elapsed / self.cfg.CALIBRATION_SECONDS, 0.0, 1.0)
        scrim = frame.copy()
        scrim[:] = (12, 16, 21)
        frame[:] = cv2.addWeighted(scrim, 0.74, frame, 0.26, 0)
        canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        cx, cy, radius = width // 2, height // 2 - 24, 48
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius),
                     outline=(66, 80, 90, 255), width=5)
        draw.arc((cx - radius, cy - radius, cx + radius, cy + radius), -90,
                 -90 + 360 * progress, fill=theme["accent"] + (255,), width=5)
        draw.text((cx, cy), f"{int(progress * 100)}%", font=fonts["strong"],
                  fill=theme["text"] + (255,), anchor="mm")
        draw.text((cx, cy + 76), "Calibrating driver - please look straight",
                  font=fonts["section"], fill=theme["text"] + (255,), anchor="mm")
        draw.text((cx, cy + 100), f"{max(0.0, self.cfg.CALIBRATION_SECONDS - elapsed):.1f}s remaining",
                  font=fonts["subtitle"], fill=theme["muted"] + (255,), anchor="mm")
        self._ui_blit(frame, canvas, 0, 0)

    # ==============================================================================
    # REDESIGNED (Part 2D / Part 3 / Part 4): professional driver-monitoring
    # console layout.
    #
    # Previous behaviour blitted the dashboard panel directly ON TOP OF the
    # right-hand pixels of the raw camera frame, so the "panel" was really
    # just a thin HUD stapled onto a full-size, dominant camera image (and it
    # covered part of the live video underneath it). That is the root cause
    # of "the live camera area is too large and covers too much of the
    # interface" - there was no true separate dashboard region at all.
    #
    # This version builds a genuinely separate composed canvas:
    #   LEFT  : the camera feed, resized DOWN to a compact, fixed width
    #           (Config.VIDEO_DISPLAY_WIDTH) with face bbox / tracking
    #           overlays (already drawn on `frame` by draw_face_box() before
    #           this runs) scaling down with it, plus a small "quick status"
    #           strip below the video (direction / tracking / confidence).
    #   RIGHT : a wide, dedicated Driver Safety Monitor panel (Config.PANEL_
    #           WIDTH) that never overlaps a single camera pixel, with all
    #           the sections requested in Part 3, plus the single "TRIP
    #           DETAILS" button from Part 4.
    #
    # The method returns the new composed canvas; run() now shows THAT
    # (instead of the raw camera `frame`) and also runs the calibration
    # overlay on it, so the full window (camera + dashboard) dims as one
    # clean modal during calibration.
    # ==============================================================================
    def _draw_calm_dashboard(self, frame, info):
        fonts, theme = self._ui_fonts(), self._UI_THEME
        cam_h, cam_w = frame.shape[:2]

        video_w = min(self.cfg.VIDEO_DISPLAY_WIDTH, cam_w)
        video_h = max(1, int(cam_h * (video_w / float(cam_w))))
        panel_width = self.cfg.PANEL_WIDTH
        canvas_h = max(self.cfg.CANVAS_HEIGHT, video_h + 40)
        canvas_w = video_w + panel_width

        # ---- Compose the base canvas: dark background, compact video top-left --
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        canvas[:] = (16, 18, 21)
        resized_video = cv2.resize(frame, (video_w, video_h), interpolation=cv2.INTER_AREA)
        canvas[0:video_h, 0:video_w] = resized_video

        level = info["level"]
        level_color = self._ui_level_color(level)
        face_state = info.get("face_state", FaceState.TRACKING)
        head = info.get("head_info", {})
        phone = info.get("phone_info", {})
        seatbelt = info.get("seatbelt_info", {})
        yawn = info.get("yawn_info", {})
        safety = clamp(float(info.get("safety_score", 100.0)), 0.0, 100.0)
        fatigue = clamp(float(info.get("fatigue_score", 0.0)), 0.0, 100.0)
        confidence = clamp(float(info.get("confidence", 0.0)), 0.0, 1.0)
        pose = info.get("pose") or {}
        direction = head.get("direction") or "CENTER"
        direction_color = self._ui_level_color(head.get("look_away_level", "GREEN"))

        # ---- Unified PHONE status (Part B): distinguishes a phone merely
        # visible somewhere in the scene from one actually being used near the
        # driver's face, from one held to the ear (a call).
        if phone.get("phone_calling"):
            phone_status, phone_alert_color = "CALL / NEAR EAR", theme["red"]
        elif phone.get("phone_alert"):
            phone_status, phone_alert_color = "IN USE", theme["red"]
        elif phone.get("phone_detected"):
            phone_status, phone_alert_color = "DETECTED", theme["yellow"]
        elif phone.get("phone_visible"):
            phone_status, phone_alert_color = "VISIBLE", theme["yellow"]
        else:
            phone_status, phone_alert_color = "CLEAR", theme["green"]

        now_ui = time.time()
        # Alert thresholds (seconds) shown on the live timer bars.
        head_alert_s = self.cfg.HEAD_WARNING_SUSTAIN_SECONDS * (
            self.cfg.HEAD_LOW_LIGHT_SUSTAIN_FACTOR if self.low_light else 1.0)
        timers = [
            ("Head look-away", head.get("look_away_duration", 0.0) if head.get("direction") else 0.0, head_alert_s),
            ("Yawn (mouth open)", yawn.get("open_duration", 0.0), self.cfg.YAWN_MIN_DURATION_SECONDS),
            ("Phone near face",
             (now_ui - self.phone_near_face_start_time) if self.phone_near_face_start_time else 0.0,
             self.cfg.PHONE_ALERT_SECONDS),
            ("Phone at ear / call",
             (now_ui - self.phone_call_start_time) if self.phone_call_start_time else 0.0,
             self.cfg.PHONE_CALL_ALERT_SECONDS),
            ("Eyes closed", info.get("closed_duration", 0.0), self.cfg.EYE_CLOSED_ALARM_SECONDS),
        ]

        # ---- Quick-status strip under the compact video (Part 2D: "current
        # driver direction" + "small status indicators" on the LEFT) --------
        strip_top = video_h
        strip_h = canvas_h - video_h
        if strip_h > 0:
            strip = Image.new("RGBA", (video_w, strip_h), (0, 0, 0, 0))
            sdraw = ImageDraw.Draw(strip)
            sdraw.rectangle((0, 0, video_w - 1, strip_h - 1), fill=(12, 15, 19, 255))
            sx0, sx1 = 16, video_w - 16
            # tracking / direction header
            sdraw.text((sx0, 14), "TRACKING", font=fonts["small"], fill=theme["muted"] + (255,))
            sdraw.text((sx0, 33), "TRACKING" if face_state == FaceState.TRACKING else str(face_state),
                       font=fonts["big"], fill=theme["text"] + (255,))
            sdraw.text((sx1, 14), "DIRECTION", font=fonts["small"],
                       fill=theme["muted"] + (255,), anchor="ra")
            sdraw.text((sx1, 33), direction, font=fonts["big"],
                       fill=direction_color + (255,), anchor="ra")
            sdraw.line((sx0, 68, sx1, 68), fill=(58, 69, 78, 200), width=1)
            sdraw.text((sx0, 80), f"Confidence {confidence * 100:.0f}%", font=fonts["body"],
                       fill=theme["text"] + (255,))
            sdraw.text((sx1, 80), "LOW LIGHT" if self.low_light else "NORMAL LIGHT",
                       font=fonts["body"],
                       fill=(theme["yellow"] if self.low_light else theme["green"]) + (255,),
                       anchor="ra")

            # at-a-glance indicator rows
            sdraw.text((sx0, 116), "STATUS", font=fonts["section"], fill=theme["accent"] + (255,))
            quick_rows = [
                ("Eyes", "CLOSED" if info["eyes_closed"] else "OPEN",
                 theme["red"] if info["eyes_closed"] else theme["green"]),
                ("Phone", phone_status, phone_alert_color),
                ("Seat belt", seatbelt.get("status", "UNAVAILABLE"),
                 theme["green"] if seatbelt.get("status") == "WORN" else theme["muted"]),
                ("Fatigue", f"{fatigue:.0f}/100", level_color),
            ]
            qy = 140
            for qlabel, qvalue, qcolor in quick_rows:
                sdraw.text((sx0, qy), qlabel.upper(), font=fonts["small"], fill=theme["muted"] + (255,))
                sdraw.text((sx1, qy - 1), qvalue, font=fonts["body"], fill=qcolor + (255,), anchor="ra")
                qy += 26

            # live ALERT TIMERS: progress toward each alert threshold
            ty = qy + 14
            sdraw.line((sx0, ty - 8, sx1, ty - 8), fill=(58, 69, 78, 200), width=1)
            sdraw.text((sx0, ty), "ALERT TIMERS", font=fonts["section"], fill=theme["accent"] + (255,))
            ty += 28
            for tlabel, tvalue, tlimit in timers:
                if ty + 34 > strip_h - 6:
                    break
                frac = clamp(tvalue / tlimit, 0.0, 1.0) if tlimit > 0 else 0.0
                tcolor = theme["green"] if frac < 0.5 else (theme["yellow"] if frac < 1.0 else theme["red"])
                sdraw.text((sx0, ty), tlabel, font=fonts["small"], fill=theme["muted"] + (255,))
                sdraw.text((sx1, ty), f"{tvalue:.1f} / {tlimit:.1f}s", font=fonts["small"],
                           fill=(tcolor if tvalue > 0 else theme["faint"]) + (255,), anchor="ra")
                sdraw.rounded_rectangle((sx0, ty + 20, sx1, ty + 26), radius=3, fill=(40, 49, 58, 255))
                if frac > 0:
                    sdraw.rounded_rectangle((sx0, ty + 20, sx0 + max(6, int((sx1 - sx0) * frac)), ty + 26),
                                            radius=3, fill=tcolor + (255,))
                ty += 40
            self._ui_blit(canvas, strip, 0, strip_top)

        # ---- Small "LIVE FEED" chip + status chip over the compact video ---
        live = Image.new("RGBA", (min(220, video_w - 16), 30), (0, 0, 0, 0))
        live_draw = ImageDraw.Draw(live)
        live_draw.rounded_rectangle((0, 0, live.width - 1, 29), radius=9,
                                    fill=(12, 18, 23, 215), outline=theme["card_border"] + (255,))
        live_draw.ellipse((10, 10, 19, 19), fill=theme["red"] + (255,))
        live_draw.text((27, 15), "LIVE FEED", font=fonts["small"], fill=theme["text"] + (255,), anchor="lm")
        self._ui_blit(canvas, live, 10, 8)

        # ================================================================
        # RIGHT PANEL: the large Driver Safety Monitor dashboard.
        # ================================================================
        panel = self._ui_gradient((panel_width, canvas_h), theme["bg_top"], theme["bg_bottom"])
        draw = ImageDraw.Draw(panel)
        left, right = 22, panel_width - 22
        draw.rectangle((0, 0, 6, canvas_h), fill=level_color + (255,))

        def text(value, xy, font="body", color=None, anchor=None):
            draw.text(xy, str(value), font=fonts[font], fill=(color or theme["text"]) + (255,), anchor=anchor)

        def card(top, bottom, fill=None, outline=None):
            draw.rounded_rectangle((left, top, right, bottom), radius=12,
                                   fill=(fill or theme["card"]) + (238,),
                                   outline=(outline or theme["card_border"]) + (220,), width=1)

        def metric_card(top, bottom, label, value, detail, color, sub=None):
            card(top, bottom)
            text(label.upper(), (left + 14, top + 9), "small", theme["muted"])
            text(value, (left + 14, top + 27), "strong", color)
            text(detail, (right - 14, top + 31), "body", theme["faint"], "ra")
            if sub:
                text(sub, (left + 14, top + 47), "small", theme["muted"])

        def ring(cx, cy, value, color, label):
            radius = 30
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius),
                         outline=(58, 69, 78, 255), width=6)
            draw.arc((cx - radius, cy - radius, cx + radius, cy + radius), -90,
                     -90 + int(360 * value / 100.0), fill=color + (255,), width=6)
            text(f"{value:.0f}", (cx, cy - 1), "strong", theme["text"], "mm")
            text(label, (cx, cy + 44), "small", theme["muted"], "mm")

        # ---- Header (Part 3 "TOP") ------------------------------------------------
        text("DRIVER MONITOR", (left, 14), "title")
        text("SAFETY CONSOLE", (left, 46), "small", theme["accent"])
        text(f"{info['fps']:.0f} FPS", (right, 18), "body", theme["faint"], "ra")
        draw.ellipse((right - 52, 49, right - 44, 57), fill=theme["green"] + (255,))
        text("LIVE", (right, 46), "small", theme["green"], "ra")
        draw.line((left, 68, right, 68), fill=theme["card_border"] + (180,), width=1)

        # ---- Status banner: active warnings flash so they cannot be missed -----------
        status = info["main_message"] if face_state != FaceState.TRACKING else {
            "GREEN": "ALL CLEAR", "YELLOW": "STAY ALERT", "ORANGE": "WARNING", "RED": "DANGER"
        }.get(level, "UNKNOWN")
        banner_fill = level_color
        if level == "RED" and int(now_ui * 2) % 2 == 1:
            banner_fill = tuple(min(255, int(c * 0.72)) for c in level_color)   # 2 Hz pulse
        draw.rounded_rectangle((left, 80, right, 126), radius=11, fill=banner_fill + (255,),
                               outline=((255, 255, 255, 235) if level in ("ORANGE", "RED") else (255, 255, 255, 50)),
                               width=3 if level in ("ORANGE", "RED") else 1)
        text(status, (left + 16, 103), "big", (18, 22, 25), "lm")
        text(level, (right - 16, 103), "big", (18, 22, 25), "rm")

        # ---- Main status: Safety / Fatigue rings + Confidence / Calibration -------
        card(138, 232)
        ring(left + 70, 172, safety, theme["green"] if safety >= 70 else theme["yellow"], "SAFETY")
        ring(left + 196, 172, fatigue, level_color, "FATIGUE")
        draw.line((left + 133, 150, left + 133, 214), fill=theme["card_border"] + (180,), width=1)
        text("CONFIDENCE", (right - 14, 150), "small", theme["muted"], "ra")
        text(f"{confidence * 100:.0f}%", (right - 14, 168), "big", theme["text"], "ra")
        text("CALIBRATION", (right - 14, 196), "small", theme["muted"], "ra")
        text("RUNNING" if self.calibrating else "READY", (right - 14, 212), "body",
             theme["accent"] if self.calibrating else theme["green"], "ra")

        # ---- EYES (Part 3) ---------------------------------------------------------
        metric_card(
            244, 314, "Eyes",
            "CLOSED" if info["eyes_closed"] else "OPEN",
            f"EAR {info['avg_ear']:.3f}",
            theme["red"] if info["eyes_closed"] else theme["green"],
            sub=f"Blinks {info.get('blink_total', 0)}  |  {info.get('blink_rate', 0.0):.0f}/min  |  "
                f"Closure {info.get('closed_duration', 0.0):.1f}s",
        )

        # ---- HEAD (Part 3): direction + yaw/pitch/roll + look-away duration --------
        head_warn = head.get("look_away_level", "GREEN") != "GREEN" and head.get("direction")
        card(324, 408, outline=direction_color if head_warn else None)
        text("HEAD", (left + 14, 333), "small", theme["muted"])
        text(direction, (left + 14, 351), "strong", direction_color)
        look_away = head.get("look_away_duration", 0.0) if head.get("direction") else 0.0
        text(f"Look-away {look_away:.1f}s / {head_alert_s:.1f}s", (right - 14, 355), "body", theme["faint"], "ra")
        text(f"Yaw {pose.get('rel_yaw_raw', pose.get('rel_yaw', 0.0)):+.0f}°     "
             f"Pitch {pose.get('rel_pitch_raw', pose.get('rel_pitch', 0.0)):+.0f}°     "
             f"Roll {pose.get('roll', 0.0):+.0f}°",
             (left + 14, 376), "body", theme["text"])
        draw.rounded_rectangle((left + 14, 398, right - 14, 402), radius=2, fill=(40, 49, 58, 255))
        if look_away > 0:
            lf = clamp(look_away / head_alert_s, 0.0, 1.0)
            draw.rounded_rectangle((left + 14, 398, left + 14 + max(5, int((right - left - 28) * lf)), 402),
                                   radius=2, fill=direction_color + (255,))

        # ---- DROWSINESS (Part 3) ----------------------------------------------------
        card(418, 494)
        text("DROWSINESS", (left + 14, 427), "small", theme["muted"])
        text(f"Fatigue {fatigue:.0f}", (left + 14, 445), "strong", level_color)
        text(f"PERCLOS {info.get('perclos', 0.0) * 100:.0f}%", (right - 14, 449), "body",
             theme["faint"], "ra")
        yawn_active = yawn.get("status", "NORMAL") != "NORMAL"
        text(f"Yawn count {yawn.get('yawn_total', self.yawn_total)}   |   "
             f"Mouth {yawn.get('status', 'NORMAL')}", (left + 14, 470), "body",
             theme["red"] if yawn.get("status") == "CONFIRMED" else (
                 theme["yellow"] if yawn_active else theme["text"]))

        # ---- PHONE + SEAT BELT side-by-side (Part 3) --------------------------------
        half = (right - left - 12) / 2.0
        p_left, p_right = left, left + half
        s_left, s_right = left + half + 12, right
        phone_hot = phone.get("phone_calling") or phone.get("phone_alert")
        draw.rounded_rectangle((p_left, 504, p_right, 570), radius=12,
                               fill=theme["card"] + (238,),
                               outline=(theme["red"] if phone_hot else theme["card_border"]) + (220,), width=1)
        draw.rounded_rectangle((s_left, 504, s_right, 570), radius=12,
                               fill=theme["card"] + (238,), outline=theme["card_border"] + (220,), width=1)
        text("PHONE", (p_left + 14, 513), "small", theme["muted"])
        text(phone_status, (p_left + 14, 531), "strong", phone_alert_color)
        text("near ear: " + ("YES" if phone.get("phone_near_ear") else "no"),
             (p_left + 14, 551), "small", theme["faint"])
        belt_value = seatbelt.get("status", "UNAVAILABLE")
        belt_color = theme["green"] if belt_value == "WORN" else (
            theme["muted"] if belt_value in ("UNAVAILABLE", "UNKNOWN") else theme["yellow"])
        text("SEAT BELT", (s_left + 14, 513), "small", theme["muted"])
        text(belt_value, (s_left + 14, 531), "strong", belt_color)

        # ---- SAFETY summary (Part 3) -------------------------------------------------
        card(580, 646)
        text("SAFETY SUMMARY", (left + 14, 588), "small", theme["muted"])
        text(f"Distraction events {len(self.distraction_events)}", (left + 14, 608), "body", theme["text"])
        text(f"Head alerts {self.head_alert_total}", (left + 14, 626), "body", theme["text"])
        text(f"Phone usage {self.phone_usage_count}", (right - 14, 608), "body", theme["text"], "ra")
        text(f"Yawns {self.yawn_total}", (right - 14, 626), "body", theme["text"], "ra")

        # ---- ALERTS (Part 3 / Part 5): up to 3 lines, fitted to the width by pixels ---
        alerts = info.get("alerts", [])
        alert_color = theme["red"] if alerts else theme["green"]
        draw.rounded_rectangle((left, 656, right, 736), radius=10, fill=(alert_color + (46,)),
                               outline=alert_color + (255,), width=2 if alerts else 1)
        draw.ellipse((left + 14, 664, left + 23, 673), fill=alert_color + (255,))
        text(f"ACTIVE ALERTS ({len(alerts)})" if alerts else "ALERTS", (left + 32, 669), "small",
             alert_color, "lm")
        shown = alerts[:3] if alerts else ["No active alerts"]
        max_text_w = (right - left) - 32 - 14
        for line_no, line in enumerate(shown):
            line = str(line)
            if draw.textlength(line, font=fonts["small"]) > max_text_w:
                while len(line) > 1 and draw.textlength(line + "…", font=fonts["small"]) > max_text_w:
                    line = line[:-1]
                line = line.rstrip() + "…"
            text(line, (left + 32, 689 + line_no * 17), "small", theme["text"], "lm")

        # ---- "TRIP DETAILS" button (Part 4): the ONE clearly visible button --------
        btn_top, btn_bottom = 746, 784
        btn_fill = theme["accent"] + (255,) if not self.show_details else level_color + (255,)
        draw.rounded_rectangle((left, btn_top, right, btn_bottom), radius=11, fill=btn_fill,
                               outline=(255, 255, 255, 60), width=1)
        btn_label = "HIDE DETAILS ▲" if self.show_details else "▤  FULL TRIP DETAILS"
        text(btn_label, ((left + right) // 2, (btn_top + btn_bottom) // 2), "strong",
             (16, 18, 20), "mm")
        # Store the button rect in CANVAS coordinates (panel is offset by
        # video_w on the composed canvas) so the mouse callback can hit-test
        # a raw click directly against it.
        self._details_button_rect = (video_w + left, btn_top, video_w + right, btn_bottom)

        text("q quit  |  r reset  |  click button or press d for full details",
             (left, canvas_h - 8), "small", theme["faint"], "lm")

        self._ui_blit(canvas, panel, video_w, 0)
        return canvas

    # ==============================================================================
    # NEW (Part 4): draw_details_panel() - the secondary "FULL DETAILS" view.
    # Built as its own image (independent of the compact main dashboard) and
    # shown in a second OpenCV window only while self.show_details is True, so
    # the main console stays visually clean as required by Part 4.
    # ==============================================================================
    def draw_details_panel(self, info):
        fonts, theme = self._ui_fonts(), self._UI_THEME
        # FIX: this panel has many sections (Part 4 lists 10+), and a fixed
        # height previously cut off / overlapped the later ones (Phone, Seat
        # Belt, Trip, Alert History). Render into a generously tall canvas
        # first, track the actual bottom of the drawn content, then crop down
        # to exactly that - so it always fits, however many rows there are.
        width, height = 1060, 1400   # two columns (was one tall 620 px column)
        img = self._ui_gradient((width, height), theme["bg_top"], theme["bg_bottom"])
        draw = ImageDraw.Draw(img)
        left, right = 20, width - 20
        y = 20

        level = info["level"]
        level_color = self._ui_level_color(level)
        head = info.get("head_info", {})
        phone = info.get("phone_info", {})
        seatbelt = info.get("seatbelt_info", {})
        yawn = info.get("yawn_info", {})
        pose = info.get("pose") or {}

        def text(value, xy, font="body", color=None, anchor=None):
            draw.text(xy, str(value), font=fonts[font], fill=(color or theme["text"]) + (255,), anchor=anchor)

        def section(title, y):
            draw.rounded_rectangle((left, y - 4, right, y + 22), radius=6, fill=theme["card"] + (235,))
            text(title, (left + 10, y), "section", theme["accent"])
            return y + 34

        def row(label, value, y, color=None):
            text(label, (left + 6, y), "body", theme["muted"])
            value = str(value)
            max_w = (right - left) - 190          # keep clear of the label column
            if draw.textlength(value, font=fonts["body"]) > max_w:
                while len(value) > 1 and draw.textlength(value + "…", font=fonts["body"]) > max_w:
                    value = value[:-1]
                value = value.rstrip() + "…"
            text(value, (right - 6, y), "body", color, "ra")
            return y + 25

        text("TRIP DETAILS", (left, y), "title")
        text(f"Session {self.session_start_time_str()}", (right, y + 4), "small", theme["faint"], "ra")
        y += 34
        # Two columns: DRIVER..DISTRACTION on the left, SAFETY..ALERT HISTORY on the right.
        # (row()/section() read left/right at call time, so re-pointing them switches column.)
        left, right = 20, 520
        title_bottom = y

        y = section("DRIVER", y)
        y = row("Face state", str(info.get("face_state", "UNKNOWN")), y)
        y = row("Face confidence", f"{clamp(float(info.get('confidence', 0.0)), 0.0, 1.0) * 100:.0f}%", y)
        y = row("Calibration status", "RUNNING" if self.calibrating else "COMPLETE", y)

        y = section("EYES", y + 6)
        y = row("EAR", f"{info.get('avg_ear', 0.0):.3f}  (L {info.get('left_ear', 0.0):.3f} / "
                        f"R {info.get('right_ear', 0.0):.3f})", y)
        y = row("Eye state", "CLOSED" if info.get("eyes_closed") else "OPEN", y,
                theme["red"] if info.get("eyes_closed") else theme["green"])
        y = row("Blink count", self.blink_total, y)
        y = row("Blink rate", f"{info.get('blink_rate', 0.0):.1f} / min", y)
        y = row("Eye closure duration", f"{info.get('closed_duration', 0.0):.1f}s", y)
        y = row("PERCLOS", f"{info.get('perclos', 0.0) * 100:.0f}%", y)

        y = section("MOUTH", y + 6)
        y = row("MAR", f"{yawn.get('mar', 0.0):.2f}", y)
        y = row("Yawn count", yawn.get("yawn_total", self.yawn_total), y)
        y = row("Mouth state", yawn.get("status", "NORMAL"), y)
        y = row("Inner-lip opening", f"{yawn.get('inner_ratio', 0.0):.2f}", y)
        y = row("Mouth-open timer", f"{yawn.get('open_duration', 0.0):.1f}s / "
                                    f"{self.cfg.YAWN_MIN_DURATION_SECONDS:.1f}s", y)

        y = section("HEAD", y + 6)
        y = row("Yaw", f"{pose.get('rel_yaw_raw', pose.get('rel_yaw', 0.0)):+.1f}°", y)
        y = row("Pitch", f"{pose.get('rel_pitch_raw', pose.get('rel_pitch', 0.0)):+.1f}°", y)
        y = row("Roll", f"{pose.get('roll', 0.0):+.1f}°", y)
        y = row("Direction", head.get("direction") or "CENTER", y,
                self._ui_level_color(head.get("look_away_level", "GREEN")))
        y = row("Look-away duration", f"{head.get('look_away_duration', 0.0):.1f}s / "
                                      f"{self.cfg.HEAD_WARNING_SUSTAIN_SECONDS:.1f}s", y)
        y = row("Sudden-turn events", "YES" if head.get("sudden_turn") else "no", y)

        y = section("DISTRACTION", y + 6)
        y = row("Distraction events", len(self.distraction_events), y)
        y = row("Head alerts (total)", self.head_alert_total, y)
        y = row("Phone events (total)", self.phone_usage_count, y)

        y_left_end = y
        left, right = 540, width - 20          # ---- switch to the right-hand column ----
        y = title_bottom

        y = section("SAFETY", y)
        y = row("Safety score", f"{clamp(float(info.get('safety_score', 100.0)), 0.0, 100.0):.0f}", y)
        y = row("Fatigue score", f"{clamp(float(info.get('fatigue_score', 0.0)), 0.0, 100.0):.0f}", y)
        y = row("Warning level", level, y, level_color)

        y = section("PHONE", y + 6)
        phone_status = ("CALL / NEAR EAR" if phone.get("phone_calling") else
                         "IN USE" if phone.get("phone_alert") else
                         "DETECTED" if phone.get("phone_detected") else
                         "VISIBLE" if phone.get("phone_visible") else "CLEAR")
        y = row("Phone status", phone_status, y)
        y = row("Phone usage count", self.phone_usage_count, y)
        y = row("Near-ear / call status", "YES" if phone.get("phone_near_ear") else "no", y)
        y = row("Phone timers (face / ear)",
                f"{(time.time() - self.phone_near_face_start_time) if self.phone_near_face_start_time else 0.0:.1f}s / "
                f"{(time.time() - self.phone_call_start_time) if self.phone_call_start_time else 0.0:.1f}s", y)

        y = section("SEAT BELT", y + 6)
        y = row("Seat-belt status", seatbelt.get("status", "UNAVAILABLE"), y)
        y = row("Currently alerting", "YES" if seatbelt.get("alert") else "no", y)

        y = section("TRIP", y + 6)
        trip_seconds = time.time() - self.session_start_time
        y = row("Trip duration", f"{int(trip_seconds // 60)}m {int(trip_seconds % 60)}s", y)
        y = row("FPS", f"{info.get('fps', 0.0):.0f}", y)
        y = row("Total active alerts (now)", len(info.get("alerts", [])), y)

        y = section("ALERT HISTORY (most recent)", y + 6)
        recent = list(self.alert_history)[-6:][::-1]
        if not recent:
            y = row("(none yet)", "", y)
        else:
            for ts, reason in recent:
                ago = max(0.0, time.time() - ts)
                y = row(f"{ago:.0f}s ago", reason, y)

        y = max(y, y_left_end) + 10
        left, right = 20, width - 20
        draw.line((left, y, right, y), fill=theme["card_border"] + (200,), width=1)
        y += 14
        text("Click the button on the main console (or press d) to close this panel.",
             (left, y), "small", theme["faint"])
        y += 30

        final_height = min(height, y)
        img = img.crop((0, 0, width, final_height))
        rgb = np.array(img.convert("RGB"))[:, :, ::-1]
        return np.ascontiguousarray(rgb)

    def session_start_time_str(self):
        try:
            return time.strftime("%H:%M:%S", time.localtime(self.session_start_time))
        except Exception:
            return "--:--:--"

    # ==============================================================================
    # NEW (Part 4): mouse callback for the "TRIP DETAILS" button. Registered
    # on the main window in run(). A click is only recognised as hitting the
    # button if it falls within the rect last drawn by _draw_calm_dashboard
    # this frame, so the hit-box always matches what's on screen even as the
    # button's fill color changes (accent -> level color) when toggled.
    # ==============================================================================
    def _on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        rect = self._details_button_rect
        if rect is None:
            return
        x1, y1, x2, y2 = rect
        if x1 <= x <= x2 and y1 <= y <= y2:
            self.show_details = not self.show_details

    # Existing OpenCV fallback remains below; calm rendering is selected above.
    def _legacy_draw_face_box(self, frame, bbox, level):
        if bbox is None:
            return
        x1, y1, x2, y2 = [int(v) for v in bbox]
        color = self.level_color(level)
        thickness = 2
        corner_len = int(min(x2 - x1, y2 - y1) * 0.18)
        corner_len = max(corner_len, 10)

        # Corner-style tracking box
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, thickness)

        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, thickness)

        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, thickness)

        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, thickness)

    # ==============================================================================
    # NEW (Feature 3): draw_calibration_overlay() - shown for the first
    # CALIBRATION_SECONDS of the run while baseline yaw/pitch/EAR are captured.
    # ==============================================================================
    def draw_calibration_overlay(self, frame, elapsed):
        self._draw_calm_calibration(frame, elapsed)
        return

        h, w = frame.shape[:2]
        remaining = max(0.0, self.cfg.CALIBRATION_SECONDS - elapsed)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (10, 10, 10), -1)
        frame[:] = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

        msg1 = "Calibrating Driver... Please Look Straight"
        msg2 = f"{remaining:.1f}s remaining"
        (tw1, th1), _ = cv2.getTextSize(msg1, self.cfg.FONT, 0.9, 2)
        (tw2, th2), _ = cv2.getTextSize(msg2, self.cfg.FONT, 0.7, 2)
        cv2.putText(frame, msg1, ((w - tw1) // 2, h // 2 - 10), self.cfg.FONT, 0.9,
                    (0, 210, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, msg2, ((w - tw2) // 2, h // 2 + 30), self.cfg.FONT, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)

        # Progress bar
        bar_w = 320
        bar_x1 = (w - bar_w) // 2
        bar_y1 = h // 2 + 55
        progress = clamp(elapsed / self.cfg.CALIBRATION_SECONDS, 0.0, 1.0)
        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x1 + bar_w, bar_y1 + 14), (90, 90, 90), 1)
        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x1 + int(bar_w * progress), bar_y1 + 14),
                      (0, 210, 255), -1)

    # ==============================================================================
    # MODIFIED: draw_dashboard()
    # Adds: current head direction with a live look-away timer, ORANGE status
    # banner support, and explicit DISTRACTED DRIVER / SUDDEN HEAD TURN /
    # INSTANT ALERT lines surfaced from head_info.
    # ==============================================================================
    def draw_dashboard(self, frame, info):
        # FIX (Part 2D): now returns the composed canvas (compact camera +
        # large dashboard panel) instead of mutating `frame` in place - see
        # _draw_calm_dashboard()'s docstring for why.
        return self._draw_calm_dashboard(frame, info)

        h, w = frame.shape[:2]
        panel_w = self.cfg.PANEL_WIDTH
        overlay = frame.copy()

        cv2.rectangle(overlay, (w - panel_w, 0), (w, h), (18, 18, 18), -1)
        alpha = 0.72
        frame[:] = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        x0 = w - panel_w + 18
        y = 34
        line_h = 26

        def put(text, color=(230, 230, 230), scale=0.58, thick=1, bold_y=None):
            nonlocal y
            yy = bold_y if bold_y is not None else y
            # NEW (Feature 10): never draw past the bottom hint bar - prevents
            # text from overlapping/spilling off the panel on smaller frames.
            if yy > h - 30:
                if bold_y is None:
                    y += line_h
                return
            cv2.putText(frame, text, (x0, yy), self.cfg.FONT, scale, color, thick, cv2.LINE_AA)
            if bold_y is None:
                y += line_h

        head_info = info["head_info"]
        level = info["level"]
        level_color = self.level_color(level)

        cv2.putText(frame, "DRIVER MONITORING SYSTEM", (x0, y), self.cfg.FONT, 0.62,
                    (255, 255, 255), 2, cv2.LINE_AA)
        y += 14
        cv2.line(frame, (x0, y), (w - 18, y), (90, 90, 90), 1)
        y += 28

        # Status banner
        cv2.rectangle(frame, (x0 - 10, y - 22), (w - 18, y + 8), level_color, -1)
        # FIX (Phase 1, Fix #2): driven solely by face_state - no more hardcoded
        # string list to keep in sync. Any non-TRACKING state's banner text is
        # main_message, which run() derives ONLY from self.face_sm.state - so a
        # newly added state (e.g. OCCLUDED) is automatically handled correctly
        # here with zero changes required in this function.
        face_state = info.get("face_state", FaceState.TRACKING)
        if face_state != FaceState.TRACKING:
            status_text = info["main_message"]
        else:
            status_text = {
                "GREEN": "STATUS: NORMAL",
                "YELLOW": "STATUS: CAUTION",
                "ORANGE": "STATUS: WARNING",
                "RED": "STATUS: DANGER",
            }.get(level, "STATUS: UNKNOWN")
        text_color = (0, 0, 0) if level in ("GREEN", "YELLOW") else (255, 255, 255)
        cv2.putText(frame, status_text, (x0, y), self.cfg.FONT, 0.62, text_color, 2, cv2.LINE_AA)
        y += 36

        put(f"FPS: {info['fps']:.1f}")
        # FIX (Phase 1, Fix #2): "Face Detected", "Face State", and the no-face
        # timer are now ALL derived from this single face_state value, so they
        # can never disagree with each other or with the banner above (the old
        # bug: face_present=True + face_stable=False during OCCLUDED used to
        # print "STABILIZING..." even though the driver was occluded, not
        # stabilizing).
        face_detected = face_state not in (FaceState.SEARCHING, FaceState.FACE_LOST)
        put(f"Face Detected: {'YES' if face_detected else 'NO'}",
            (60, 200, 60) if face_detected else (0, 0, 255))
        face_state_colors = {
            FaceState.STABILIZING: (0, 210, 255),
            FaceState.OCCLUDED: (0, 165, 255),
            FaceState.SEARCHING: (0, 210, 255),
            FaceState.FACE_LOST: (0, 0, 255),
        }
        if face_state != FaceState.TRACKING:
            put(f"Face State: {face_state}", face_state_colors.get(face_state, (200, 200, 200)))
        if face_state in (FaceState.SEARCHING, FaceState.FACE_LOST) and info.get("no_face_duration", 0.0) > 0:
            put(f"No-Face Timer: {info['no_face_duration']:.1f}s", (0, 0, 255))
        # NEW (Feature 10): calibration + low-light status always visible
        calib_color = (0, 210, 255) if self.calibrating else (60, 200, 60)
        put(f"Calibration: {'IN PROGRESS' if self.calibrating else 'DONE'}", calib_color)
        if self.low_light:
            put("Lighting: LOW LIGHT (enhanced)", (0, 165, 255))
        y += 4
        cv2.line(frame, (x0, y), (w - 18, y), (70, 70, 70), 1)
        y += 22

        # NEW (Issue 8 dashboard): Safety Score, Confidence, Emotion placeholder
        cv2.putText(frame, "DRIVER STATUS", (x0, y), self.cfg.FONT, 0.55, (0, 210, 255), 2, cv2.LINE_AA)
        y += line_h
        safety_score = info.get("safety_score", 100.0)
        safety_color = (60, 200, 60) if safety_score >= 70 else (
            (0, 210, 255) if safety_score >= 45 else (0, 0, 255))
        s_bar_x1, s_bar_y1 = x0, y
        s_bar_x2, s_bar_y2 = w - 18, y + 16
        cv2.rectangle(frame, (s_bar_x1, s_bar_y1), (s_bar_x2, s_bar_y2), (90, 90, 90), 1)
        s_fill_w = int((s_bar_x2 - s_bar_x1) * (safety_score / 100.0))
        cv2.rectangle(frame, (s_bar_x1, s_bar_y1), (s_bar_x1 + s_fill_w, s_bar_y2), safety_color, -1)
        cv2.putText(frame, f"Safety Score: {safety_score:.0f}/100", (s_bar_x1 + 4, s_bar_y2 - 3),
                    self.cfg.FONT, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
        y += 28
        confidence = info.get("confidence", 0.0)
        conf_color = (60, 200, 60) if confidence >= self.cfg.MIN_POSE_CONFIDENCE else (
            (0, 210, 255) if confidence >= 0.5 else (0, 0, 255))
        put(f"Confidence: {confidence * 100:.0f}%", conf_color)
        # Emotion recognition is not part of this build - shown as an explicit
        # placeholder rather than a fabricated value.
        put("Emotion: N/A (not available)", (150, 150, 150))
        y += 4
        cv2.line(frame, (x0, y), (w - 18, y), (70, 70, 70), 1)
        y += 22

        cv2.putText(frame, "EYE METRICS", (x0, y), self.cfg.FONT, 0.55, (0, 210, 255), 2, cv2.LINE_AA)
        y += line_h
        put(f"EAR (avg): {info['avg_ear']:.3f}")
        put(f"EAR L/R: {info.get('left_ear', 0.0):.3f} / {info.get('right_ear', 0.0):.3f}", scale=0.5)
        ec_color = (0, 0, 255) if info["eyes_closed"] else (60, 200, 60)
        put(f"Eyes: {'CLOSED' if info['eyes_closed'] else 'OPEN'}", ec_color)
        put(f"Closed Duration: {info['closed_duration']:.2f}s")
        put(f"Blink Count: {info['blink_total']}")
        put(f"Blink Rate: {info['blink_rate']:.1f}/min")
        # NEW (Feature 2): MAR / yawn readout
        yawn_info = info.get("yawn_info", {})
        yawn_color = (0, 0, 255) if yawn_info.get("yawning") else (230, 230, 230)
        put(f"MAR: {yawn_info.get('mar', 0.0):.3f}", yawn_color)
        put(f"Yawn Count: {yawn_info.get('yawn_total', 0)}")
        if yawn_info.get("yawning"):
            put("YAWN DETECTED", (0, 0, 255))
        y += 4
        cv2.line(frame, (x0, y), (w - 18, y), (70, 70, 70), 1)
        y += 22

        cv2.putText(frame, "HEAD POSE", (x0, y), self.cfg.FONT, 0.55, (0, 210, 255), 2, cv2.LINE_AA)
        y += line_h
        if info["pose"] is not None:
            put(f"Yaw: {info['pose']['yaw']:.1f} deg (rel {info['pose']['rel_yaw']:+.1f})")
            put(f"Pitch: {info['pose']['pitch']:.1f} deg (rel {info['pose']['rel_pitch']:+.1f})")
            put(f"Roll: {info['pose']['roll']:.1f} deg")
        else:
            put("Pose: N/A")

        # NEW: current head direction + live look-away timer, e.g. "LEFT (1.8 sec)"
        direction_text = head_info["direction"] if head_info["direction"] else "CENTER"
        if head_info["direction"] and head_info["look_away_duration"] > 0:
            direction_display = f"{direction_text} ({head_info['look_away_duration']:.1f} sec)"
        else:
            direction_display = direction_text
        direction_color = self.level_color(head_info["look_away_level"]) \
            if head_info["direction"] else (60, 200, 60)
        put(f"Head Direction: {direction_display}", direction_color)
        y += 4
        cv2.line(frame, (x0, y), (w - 18, y), (70, 70, 70), 1)
        y += 22

        cv2.putText(frame, "FATIGUE SCORE", (x0, y), self.cfg.FONT, 0.55, (0, 210, 255), 2, cv2.LINE_AA)
        y += line_h
        score = info["fatigue_score"]
        bar_x1, bar_y1 = x0, y
        bar_x2, bar_y2 = w - 18, y + 18
        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (90, 90, 90), 1)
        fill_w = int((bar_x2 - bar_x1) * (score / 100.0))
        bar_color = self.level_color(level)
        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x1 + fill_w, bar_y2), bar_color, -1)
        cv2.putText(frame, f"{score:.0f}/100", (bar_x1 + 6, bar_y2 - 4), self.cfg.FONT, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        y += 34
        put(f"PERCLOS: {info['perclos']:.1f}%")
        y += 4
        cv2.line(frame, (x0, y), (w - 18, y), (70, 70, 70), 1)
        y += 22

        # NEW (Features 4/5): PHONE & SEAT BELT status
        cv2.putText(frame, "PHONE / SEAT BELT", (x0, y), self.cfg.FONT, 0.55, (0, 210, 255), 2, cv2.LINE_AA)
        y += line_h
        phone_info = info.get("phone_info", {})
        if self.cfg.ENABLE_PHONE_DETECTION and self.phone_detector.is_available():
            phone_color = (0, 0, 255) if phone_info.get("phone_alert") else (60, 200, 60)
            put(f"Phone: {'DETECTED NEAR FACE' if phone_info.get('phone_detected') else 'not detected'}",
                phone_color)
        else:
            put("Phone detection: OFF", (150, 150, 150))

        seatbelt_info = info.get("seatbelt_info", {})
        sb_status = seatbelt_info.get("status", "UNAVAILABLE")
        sb_color = {
            "WORN": (60, 200, 60), "NOT WORN": (0, 0, 255),
        }.get(sb_status, (150, 150, 150))
        put(f"Seat Belt: {sb_status}", sb_color)
        y += 4
        cv2.line(frame, (x0, y), (w - 18, y), (70, 70, 70), 1)
        y += 22

        cv2.putText(frame, "ACTIVE ALERTS", (x0, y), self.cfg.FONT, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
        y += line_h
        current_alert = info["alerts"][0] if info["alerts"] else "None"
        put(f"Current Alert: {current_alert}", level_color, scale=0.5)
        if info["alerts"]:
            for alert_text in info["alerts"][:8]:
                put(f"! {alert_text}", (0, 0, 255), scale=0.5)
        else:
            put("No active alerts", (60, 200, 60), scale=0.52)

        # Bottom hint bar
        cv2.putText(frame, "Press 'q' to quit  |  'r' to reset", (x0, h - 16),
                    self.cfg.FONT, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

        # Top-left main status message (large, on video area)
        main_msg, main_color = info["main_message"], level_color
        cv2.rectangle(frame, (10, 10), (10 + 9 * len(main_msg) + 20, 46), (15, 15, 15), -1)
        cv2.putText(frame, main_msg, (20, 38), self.cfg.FONT, 0.78, main_color, 2, cv2.LINE_AA)

    # ------------------------------------------------------------------------------
    def update_fps(self, now):
        dt = now - self.prev_frame_time
        self.prev_frame_time = now
        if dt > 0:
            self.fps_smooth.append(1.0 / dt)
        self.fps = sum(self.fps_smooth) / len(self.fps_smooth) if self.fps_smooth else 0.0

    # ==============================================================================
    # NEW: update_alarm_and_voice()
    # ==============================================================================
    # FIX (audio audibility req #9/#10): central alert dispatcher.
    # Every audible + spoken alert in the whole project now flows through
    # trigger_alert() - it owns starting the alarm sound, the per-alert-type
    # "[Alert] X -> SOUND" diagnostic line (with its own cooldown so it
    # doesn't spam the console every frame while RED persists), and the
    # spoken voice alert. Detection logic, thresholds, timers, eye/state
    # machines, and the visual dashboard (active_alerts/banner) are NOT
    # touched by this function - it is purely the audio/voice/diagnostics
    # layer sitting on top of decisions made elsewhere, unchanged.
    # ==============================================================================
    def _determine_alert_type(self, head_info, yawn_info, phone_info, seatbelt_info,
                               face_message, level):
        """
        Labels WHICH condition is driving the current alert, using the exact
        same priority order as the message selection below - purely for the
        "[Alert] X -> SOUND" diagnostic line (req #10). Never changes which
        condition wins or when an alert fires.
        """
        if face_message:
            return "FACE"
        # FIX (req #2): phone-to-ear is a distinct, more specific condition
        # than the general "phone visible near face" - labeled separately.
        if phone_info and phone_info.get("phone_calling"):
            return "PHONE_CALLING"
        if phone_info and phone_info.get("phone_alert"):
            return "PHONE"
        if seatbelt_info and seatbelt_info.get("alert"):
            return "SEATBELT"
        # FIX (req #3/#4): sustained-confirmed-yawn (mouth still open past
        # confirmation) now drives its own RED+SOUND state via update_warning_level.
        if yawn_info and (yawn_info.get("yawning") or yawn_info.get("yawn_just_confirmed")):
            return "YAWN"
        if head_info["distracted"]:
            return "HEAD_DISTRACTION"
        # FIX (req #1/#10): a sustained head-direction RED used to fall through
        # to the generic "DROWSINESS" label below with no direction info at
        # all - now labeled per-direction (HEAD_LEFT/HEAD_RIGHT/HEAD_UP/HEAD_DOWN).
        for item in head_info.get("sustained_alerts", []):
            if item["level"] == "RED":
                return f"HEAD_{item['direction']}"
        if head_info["sudden_turn"]:
            return "HEAD_SUDDEN_TURN"
        if level == "RED":
            return "DROWSINESS"
        return "GENERAL"

    def trigger_alert(self, alert_type, message):
        """
        Central alert dispatcher (req #9): visual alerts are unaffected
        (they're already appended to active_alerts elsewhere, unchanged) -
        this handles audible beep/WAV + voice + their cooldowns + diagnostics.
        """
        now = time.time()
        last_sound_time = self._alert_type_last_sound_time.get(alert_type, 0.0)
        if now - last_sound_time >= self.cfg.ALERT_SOUND_COOLDOWN_SECONDS:
            self._alert_type_last_sound_time[alert_type] = now
            print(f"[Alert] {alert_type} -> SOUND")
        # AlarmManager.start() is a no-op if already active (single persistent
        # thread - req #7 "do not continuously create new threads").
        self.alarm.start()
        self._speak_with_cooldown(message, now)

    def _speak_with_cooldown(self, message, now):
        if self.voice is None or not message:
            return
        # Global minimum gap between ANY two spoken phrases, so alerts never
        # talk over each other/themselves.
        if now - self.last_voice_time < self.cfg.VOICE_COOLDOWN_SECONDS:
            return
        # Per-message-text cooldown - the SAME alert cannot repeat within
        # ALERT_COOLDOWN_SECONDS, independent of the global gap above.
        last_time = self._voice_message_last_time.get(message, 0.0)
        if now - last_time < self.cfg.ALERT_COOLDOWN_SECONDS:
            return
        self.voice.speak(message)
        self.last_voice_time = now
        self._voice_message_last_time[message] = now

    # ==============================================================================
    # NEW (req #5): print_debug_diagnostics() - optional console-only status
    # lines for head/phone/yawn state, throttled to DEBUG_CONSOLE_INTERVAL_SECONDS
    # so it never spams the console. This is purely for debugging - it does
    # NOT touch or replace the dashboard in any way.
    # ==============================================================================
    def print_debug_diagnostics(self, head_info, phone_info, yawn_info, now):
        if not self.cfg.DEBUG_CONSOLE_DIAGNOSTICS:
            return
        if now - self._last_debug_print_time < self.cfg.DEBUG_CONSOLE_INTERVAL_SECONDS:
            return
        self._last_debug_print_time = now

        head_direction = head_info.get("direction") or "CENTER"
        head_timer = head_info.get("look_away_duration", 0.0)
        print(f"[HEAD] direction={head_direction} distraction_timer={head_timer:.1f}")

        print(f"[PHONE] detected={phone_info.get('phone_detected', False)} "
              f"near_ear={phone_info.get('phone_near_ear', False)} "
              f"calling={phone_info.get('phone_calling', False)}")

        print(f"[YAWN] MAR={yawn_info.get('mar', 0.0):.2f} "
              f"timer={yawn_info.get('open_duration', 0.0):.1f} "
              f"confirmed={yawn_info.get('yawning', False)} "
              f"status={yawn_info.get('status', 'NORMAL')}")

    # ==============================================================================
    # Centralised alarm + voice logic (feature 8). The beep alarm now fires for
    # ANY RED-level condition (eye closure, head L/R/U/D, sudden turn, distracted
    # driver, prolonged face loss) and stays continuous while RED holds; it gives
    # a short pulse on ORANGE. Spoken phrases are throttled by VOICE_COOLDOWN_SECONDS
    # so they don't talk over themselves.
    # ==============================================================================
    def update_alarm_and_voice(self, level, head_info, now, yawn_info=None, phone_info=None,
                                seatbelt_info=None, face_message=None):
        # Message-priority logic - UNCHANGED from before this fix, with two
        # additions inserted at the appropriate priority spots: phone-calling
        # (req #2, more specific/severe than plain phone-near-face) and
        # sustained-head-direction RED (req #1, was previously unreferenced
        # here and fell through to the generic "Look forward." message).
        message = None
        if face_message:
            message = face_message
        elif phone_info and phone_info.get("phone_calling"):
            message = "Phone call detected. Please put the phone away."
        elif phone_info and phone_info.get("phone_alert"):
            message = "Phone distraction detected. Please focus on driving."
        elif seatbelt_info and seatbelt_info.get("alert"):
            message = "Please wear your seat belt."
        elif yawn_info and (yawn_info.get("yawning") or yawn_info.get("yawn_just_confirmed")):
            message = "You seem tired. Consider taking a break."
        elif head_info["distracted"]:
            message = "Driver distracted."
        elif any(item["level"] == "RED" for item in head_info.get("sustained_alerts", [])):
            red_item = next(item for item in head_info["sustained_alerts"] if item["level"] == "RED")
            message = f"Please look at the road. Head turned {red_item['direction'].lower()}."
        elif head_info["sudden_turn"]:
            message = "Please keep your eyes on the road."
        elif level == "RED":
            message = "Look forward."
        elif level == "ORANGE" and head_info.get("look_away_level") != "ORANGE":
            # FIX: the head ORANGE stage (1.5-2.0 s) is a visual lead-up to the 2.0 s
            # alert. Speaking here would also start the voice cooldown and delay the
            # actual alert's voice, so it stays silent; other ORANGE causes (drowsy
            # eyes, fused fatigue pattern) still speak exactly as before.
            message = "Please keep your eyes on the road."

        # PROBLEM 5 (UNCHANGED trigger condition): the audible siren only
        # plays at RED - eye closure alone only ever reaches RED at the
        # CRITICAL stage (>= 4.0s continuously closed) - see
        # classify_eye_closure_stage()/update_warning_level(). ORANGE still
        # gets a spoken voice nudge, just not the continuous alarm tone.
        if level == "RED":
            alert_type = self._determine_alert_type(
                head_info, yawn_info, phone_info, seatbelt_info, face_message, level)
            self.trigger_alert(alert_type, message)
        else:
            # req #7: condition becomes CLEAR -> immediately stop/reset sound.
            self.alarm.stop()
            self._speak_with_cooldown(message, now)

    # ==============================================================================
    # MODIFIED: run() - wires all new features together each frame.
    # ==============================================================================
    # ==============================================================================
    # NEW (Features 6/7): generate_trip_report() - builds the stats dict, writes
    # Trip_Report.pdf, and (if enabled) uploads the same data to Firestore.
    # Called once from the `finally` block in run() on program exit.
    # ==============================================================================
    def generate_trip_report(self):
        # NEW (Feature 11/12): never generate twice, never let a report error
        # prevent clean shutdown.
        if self._trip_report_generated:
            return
        self._trip_report_generated = True
        try:
            self._build_and_save_trip_report()
        except Exception as exc:
            print(f"[TripReport] Unexpected error while generating report: {exc}")

    def _build_and_save_trip_report(self):
        import datetime
        now = time.time()
        elapsed_seconds = now - self.session_start_time
        hh = int(elapsed_seconds // 3600)
        mm = int((elapsed_seconds % 3600) // 60)
        ss = int(elapsed_seconds % 60)
        driving_time_str = f"{hh:02d}:{mm:02d}:{ss:02d}"

        # Reuses the SAME formula as the live dashboard "Safety Score" - one
        # implementation, no duplicate scoring logic (Issue 10).
        safety_score = self.compute_live_safety_score(now)
        if safety_score >= 85:
            rating = f"A (Safe) - {safety_score:.0f}/100"
        elif safety_score >= 65:
            rating = f"B (Caution) - {safety_score:.0f}/100"
        elif safety_score >= 45:
            rating = f"C (Risky) - {safety_score:.0f}/100"
        else:
            rating = f"D (Dangerous) - {safety_score:.0f}/100"

        stats = {
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "driving_time": driving_time_str,
            "avg_fps": f"{self.fps:.1f}",
            "blink_count": self.blink_total,
            "yawn_count": self.yawn_total,
            "fatigue_score": f"{self.fatigue_score:.0f}/100",
            "phone_usage_count": self.phone_usage_count,
            "distraction_count": len(self.distraction_events),
            "head_alerts": self.head_alert_total,
            "seatbelt_status": self.seatbelt_status,
            "safety_rating": rating,
        }

        if self.cfg.ENABLE_TRIP_REPORT:
            path = self.trip_report_gen.generate(stats)
            if path:
                print(f"[TripReport] Saved: {path}")

        if self.cfg.ENABLE_FIREBASE:
            firebase_payload = dict(stats)
            firebase_payload["driver_name"] = self.cfg.DRIVER_NAME
            firebase_payload["trip_duration_seconds"] = elapsed_seconds
            self.firebase.upload_async(firebase_payload)

    # ------------------------------------------------------------------------------
    # NEW (startup diagnostics): prints a clear OK/FAILED status line for every
    # optional/required subsystem BEFORE the main loop starts, so problems are
    # obvious immediately instead of manifesting later as silently-missing
    # detections. Never raises on its own - only reports status; open_camera()
    # (called right after) is what actually aborts startup if the camera fails.
    # ------------------------------------------------------------------------------
    def print_startup_diagnostics(self):
        print("-" * 70)
        print(" STARTUP DIAGNOSTICS")
        print("-" * 70)

        # Camera - checked here informationally; open_camera() (called right
        # after this) is the actual authority and will raise if it fails.
        try:
            test_cap = cv2.VideoCapture(self.cfg.CAMERA_INDEX)
            cam_ok = test_cap.isOpened()
            test_cap.release()
        except Exception as exc:
            cam_ok = False
            print(f"[Camera] FAILED: {exc}")
        if cam_ok:
            print("[Camera] OK")
        elif not hasattr(self, "_camera_diag_printed_error"):
            print(f"[Camera] FAILED (index {self.cfg.CAMERA_INDEX} could not be opened)")

        # MediaPipe / Face Landmarker
        if getattr(self, "mediapipe_ok", False):
            print(f"[MediaPipe] OK (backend={self.face_mesh.backend})")
            print("[Face Landmarker] OK")
        else:
            print("[MediaPipe] FAILED")
            print("[Face Landmarker] FAILED")

        # YOLO - phone + seat belt detection
        yolo_lines = []
        if self.cfg.ENABLE_PHONE_DETECTION:
            if self.phone_detector.is_available():
                yolo_lines.append(f"phone: OK ({self.cfg.YOLO_PHONE_MODEL_PATH})")
            else:
                reason = self.phone_detector.load_error or "model unavailable"
                yolo_lines.append(f"phone: FAILED ({reason})")
        else:
            yolo_lines.append("phone: DISABLED")
        if self.cfg.ENABLE_SEATBELT_DETECTION:
            if self.seatbelt_detector.is_available():
                yolo_lines.append(f"seatbelt: OK ({self.cfg.YOLO_SEATBELT_MODEL_PATH})")
            else:
                reason = self.seatbelt_detector.load_error or \
                    f"model file not found: {self.cfg.YOLO_SEATBELT_MODEL_PATH}"
                yolo_lines.append(f"seatbelt: DISABLED ({reason})")
        else:
            yolo_lines.append("seatbelt: DISABLED")
        yolo_all_ok = all("FAILED" not in line for line in yolo_lines)
        print(f"[YOLO] {'OK' if yolo_all_ok else 'PARTIAL'} - {', '.join(yolo_lines)}")

        # Audio (alarm) - FIX (req #6/#14): runs a REAL test of both the WAV
        # and Beep playback paths independently before the camera starts, and
        # self_test() itself prints the exact "[Audio] TEST START" /
        # "[Audio] BEEP: OK|FAILED: ..." / "[Audio] WAV: OK|FAILED: ..."
        # lines - never assumes "enabled" means "works".
        beep_ok, wav_ok = self.alarm.self_test()
        if not wav_ok and not beep_ok:
            print("[Audio] WARNING: neither WAV nor Beep playback works - alerts will "
                  "only be visible on the dashboard, not audible.")
        elif not wav_ok:
            print("[Audio] NOTE: WAV playback unavailable - falling back to Beep only "
                  "(still audible, but WAV is the more reliable option).")

        # Voice (pyttsx3) - FIX (req #4): report the real init/speak error,
        # never just "unavailable".
        if not self.cfg.ENABLE_VOICE_ALERTS:
            print("[Voice] DISABLED (Config.ENABLE_VOICE_ALERTS=False)")
        elif self.voice is not None and self.voice.is_available():
            print("[Voice] pyttsx3: OK")
            self.voice.speak("Voice alert check.")
        else:
            err = self.voice.init_error if self.voice is not None else "not initialized"
            print(f"[Voice] pyttsx3: FAILED: {err}")

        # ReportLab (trip report) - FIX (req #12): exact wording requested,
        # and never crashes/blocks the monitoring system either way.
        try:
            import reportlab  # noqa: F401
            print("[TripReport] reportlab: OK")
        except ImportError:
            print("[TripReport] reportlab not installed - PDF disabled")

        # Firebase
        if self.cfg.ENABLE_FIREBASE:
            print(f"[Firebase] {'OK' if self.firebase.enabled else 'FAILED/DISABLED'}")
        else:
            print("[Firebase] DISABLED (Config.ENABLE_FIREBASE=False)")

        # Calibration hasn't run yet at this point in startup - report status
        # honestly (it prints its own "[Calibration] Done. ..." /
        # "[Calibration] WARNING: ..." lines once it actually completes).
        print(f"[Calibration] PENDING - will calibrate for {self.cfg.CALIBRATION_SECONDS:.0f}s "
              f"once the camera starts. Face the camera normally during this time.")

        print("-" * 70)

    # ------------------------------------------------------------------------------
    def run(self):
        self.print_startup_diagnostics()
        self.open_camera()
        window_name = "DMS | Driver Safety Monitor"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name,
                          self.cfg.VIDEO_DISPLAY_WIDTH + self.cfg.PANEL_WIDTH,
                          self.cfg.CANVAS_HEIGHT)
        # NEW (Part 4): "TRIP DETAILS" button is a real clickable region on
        # the main window - see _on_mouse()/_draw_calm_dashboard().
        cv2.setMouseCallback(window_name, self._on_mouse)

        try:
            while True:
                ret, frame = self.cap.read()

                # ---------------- NEW (Feature 11): camera disconnect handling ------
                if not ret or frame is None:
                    self._camera_fail_streak += 1
                    if self._camera_fail_streak >= self.cfg.CAMERA_READ_FAIL_ABORT_LIMIT:
                        print("[Camera] Too many consecutive read failures - "
                              "attempting to reopen the camera.")
                        if not self.reopen_camera():
                            print("[Camera] Could not recover the camera. Exiting.")
                            break
                        self._camera_fail_streak = 0
                    time.sleep(0.01)
                    continue
                self._camera_fail_streak = 0

                frame = cv2.flip(frame, 1)

                # ---------------- NEW (Feature 11): low-light detection/enhancement -
                frame, current_brightness = self.check_and_enhance_low_light(frame)

                frame_h, frame_w = frame.shape[:2]
                now = time.time()
                self.update_fps(now)

                # ---------------- NEW (perf): shared YOLO scheduling cadence --------
                self._yolo_frame_counter += 1
                run_yolo_this_frame = (self._yolo_frame_counter % self.cfg.YOLO_DETECT_EVERY_N_FRAMES == 0)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = self.face_mesh.process(rgb)

                avg_ear = 0.0
                left_ear = 0.0
                right_ear = 0.0
                eyes_closed = False
                closed_duration = 0.0
                raw_pose = None
                bbox = None
                mar = 0.0
                self.active_alerts = []

                # Default head-tracking payload when no face / no pose is present
                head_info = {
                    "direction": None,
                    "instant_alert": None,
                    "sudden_turn": False,
                    "sustained_alerts": [],
                    "look_away_duration": 0.0,
                    "look_away_level": "GREEN",
                    "distracted": self.distracted_flag,
                    "pose": None,
                }
                yawn_info = {"mar": 0.0, "mouth_open": False, "open_duration": 0.0,
                              "yawning": False, "yawn_just_confirmed": False,
                              "yawn_total": self.yawn_total}
                elapsed = 0.0
                if self.calibrating and self.calibration_start_time is not None:
                    elapsed = now - self.calibration_start_time

                # ==========================================================================
                # FIX (Phase 1 - unified face state machine): FACE-DETECTION GATING
                # self.face_sm is the ONE source of truth for face presence / stability /
                # occlusion. This block's only job is to (a) parse this frame's landmarks
                # if any, (b) run the geometric occlusion check, (c) feed both signals to
                # the state machine, and (d) run EAR/MAR/pose computation ONLY while
                # TRACKING. No other module keeps its own face_present/face_stable/timer
                # bookkeeping any more.
                # ==========================================================================
                was_tracking = self.face_sm.is_tracking
                landmarks_present = bool(results.multi_face_landmarks)
                occlusion_detected = False
                landmarks_px = None
                bbox = self.smoothed_bbox

                if landmarks_present:
                    face_landmarks = results.multi_face_landmarks[0]
                    landmarks_px = self.landmarks_to_px(face_landmarks, frame_w, frame_h)

                    # Bounding box + eye markers are purely visual/tracking aids and are
                    # safe to draw even before the face is considered "stable".
                    bbox = self.update_face_bbox(landmarks_px)
                    for idx in self.LEFT_EYE_IDX + self.RIGHT_EYE_IDX:
                        p = landmarks_px[idx]
                        cv2.circle(frame, (int(p[0]), int(p[1])), 2, (0, 255, 255), -1)

                    # NEW (Phase 1, goal #7): geometric occlusion check - see
                    # detect_face_occlusion() docstring for the reasoning. Only
                    # meaningful once bbox has stabilized somewhat, so it's safe
                    # to run every frame landmarks exist.
                    occlusion_detected = self.detect_face_occlusion(landmarks_px, bbox)

                self.face_sm.update(landmarks_present, occlusion_detected, now)
                is_tracking = self.face_sm.is_tracking

                if landmarks_present:
                    if is_tracking:
                        left_ear = eye_aspect_ratio(landmarks_px, self.LEFT_EYE_IDX)
                        right_ear = eye_aspect_ratio(landmarks_px, self.RIGHT_EYE_IDX)
                        avg_ear = (left_ear + right_ear) / 2.0
                        self.ear_history.append(avg_ear)
                        sma_ear = sum(self.ear_history) / len(self.ear_history)
                        # PROBLEM 10: EMA layered on top of the SMA (mirrors the
                        # yaw/pitch/roll smoothing) to remove residual flicker.
                        if self._ema_ear is None:
                            self._ema_ear = sma_ear
                        else:
                            self._ema_ear = (self.cfg.EAR_EMA_ALPHA * sma_ear) + \
                                             ((1 - self.cfg.EAR_EMA_ALPHA) * self._ema_ear)
                        smoothed_ear = self._ema_ear

                        raw_pose = self.estimate_head_pose(landmarks_px, frame_w, frame_h)
                        # FIX: mouth ratios are measured on the 3D landmarks (MediaPipe x, y, z),
                        # so they do not change when the head pitches or yaws - the old 2D ratio
                        # shrank or grew with head angle (a closed mouth read 0.20-0.30 across
                        # poses and an open mouth under-read when the head was down). The mouth
                        # is judged from mouth landmarks only, never from head pitch.
                        mouth_pts = self._landmarks_3d if self._landmarks_3d is not None else landmarks_px
                        mar = mouth_aspect_ratio(mouth_pts, self.MOUTH_IDX)
                        mouth_width_3d = euclidean_distance(mouth_pts[self.MOUTH_IDX[0]],
                                                            mouth_pts[self.MOUTH_IDX[1]])
                        inner_ratio = (euclidean_distance(mouth_pts[self.MOUTH_INNER_IDX[0]],
                                                          mouth_pts[self.MOUTH_INNER_IDX[1]]) / mouth_width_3d
                                       if mouth_width_3d > 0 else 0.0)

                        # ---------------- NEW (Feature 3): startup calibration -------
                        if self.calibrating:
                            elapsed = self.run_calibration_step(raw_pose, avg_ear, now, mar,
                                                                 current_brightness)
                        else:
                            eyes_closed, closed_duration = self.process_eye_state(smoothed_ear, now)
                            head_info = self.process_head_direction(raw_pose, now)
                            # ---------------- NEW (Feature 2): MAR / yawn detection ---
                            yawn_info = self.process_yawn(mar, now, inner_ratio)
                    # else: face present but state machine says not TRACKING yet
                    # (STABILIZING/OCCLUDED) - bbox/eye markers drawn above for
                    # feedback, but NO EAR/MAR/pose/blink/fatigue computed.
                else:
                    self.smoothed_bbox = None

                # ---- Smoothing-buffer reset happens ONCE, exactly on the transition
                # OUT of TRACKING (never on a single debounced frame miss - that's the
                # whole point of the state machine). This is what stops a one-frame
                # MediaPipe blip from wiping EAR/pose history and forcing a fresh
                # stabilization wait every time.
                if was_tracking and not is_tracking:
                    self.prev_yaw = None
                    self.prev_pitch = None
                    self.prev_pose_time = None
                    self._instant_streak = 0
                    for k in self._head_alert_latched:
                        self._head_alert_latched[k] = False
                    self.ear_history.clear()
                    self._ema_ear = None
                    self.mar_hist.clear()
                    self.inner_hist.clear()
                    # FIX: stale timers must not survive a tracking gap - on re-acquisition
                    # they would make a head turn / open mouth look "already sustained".
                    for k in self.direction_start_time:
                        self.direction_start_time[k] = None
                    self._last_direction = None
                    self._pnp_prev = None
                    self.mouth_open_start_time = None
                    self._mouth_last_open_time = None
                    self.yawn_confirmed_for_current_event = False
                    self.yawning_now = False
                    self.yaw_smooth_hist.clear()
                    self.pitch_smooth_hist.clear()
                    self.roll_smooth_hist.clear()
                    self._ema_yaw = None
                    self._ema_pitch = None
                    self._ema_roll = None
                    self._bbox_size_hist.clear()   # NEW (Phase 1, Fix #4): stale bbox jitter is meaningless post-exit

                # ---------------- NEW (Features 4/5): phone & seat-belt detection ---
                # Only run once calibrated AND the face is stable - a phone/seat-belt
                # check against a bbox from an unstable/absent face is meaningless.
                if not self.calibrating and is_tracking:
                    phone_info = self.process_phone_detection(frame, bbox, now, run_yolo_this_frame)
                    # FIX (req #2): reuses the YOLO boxes phone_info just computed
                    # (self._last_phone_boxes / phone_info["boxes"]) - no duplicate
                    # YOLO inference - combined with face landmarks (or a safe bbox-
                    # edge fallback) to distinguish PHONE DETECTED from PHONE CALLING.
                    frame_area = frame.shape[0] * frame.shape[1]
                    call_info = self.process_phone_call_detection(
                        phone_info.get("boxes_all", phone_info.get("boxes", [])),
                        landmarks_px, bbox, frame_area, now)
                    phone_info.update(call_info)

                    # Draw what the detector actually sees (phone boxes + the ear reference
                    # points) so detection problems are visible instead of silent. Pure
                    # drawing on the camera frame - no effect on any decision.
                    if phone_info.get("boxes_all"):
                        for (bx1, by1, bx2, by2, _bc, bconf) in phone_info["boxes_all"]:
                            if phone_info.get("phone_near_ear"):
                                box_color = (60, 60, 240)       # red: at the ear
                            elif phone_info.get("phone_detected"):
                                box_color = (0, 190, 255)       # amber: near the face
                            else:
                                box_color = (190, 190, 190)     # grey: visible, not near the driver
                            cv2.rectangle(frame, (int(bx1), int(by1)), (int(bx2), int(by2)), box_color, 3)
                            # sized for the camera feed being shown at roughly half width
                            cv2.putText(frame, f"PHONE {bconf:.2f}", (int(bx1), max(22, int(by1) - 8)),
                                        self.cfg.FONT, 0.9, box_color, 2, cv2.LINE_AA)
                        if landmarks_px is not None:
                            for ear_idx in (self.LEFT_EAR_LANDMARK_IDX, self.RIGHT_EAR_LANDMARK_IDX):
                                ep = landmarks_px[ear_idx]
                                cv2.circle(frame, (int(ep[0]), int(ep[1])), 6, (255, 200, 0), -1)
                    seatbelt_info = self.process_seatbelt_detection(frame, now, run_yolo_this_frame)
                else:
                    phone_info = {"phone_detected": False, "phone_alert": False, "boxes": [],
                                  "boxes_all": [],
                                  "phone_near_ear": False, "phone_calling": False,
                                  "phone_visible": False}
                    seatbelt_info = {"status": self.seatbelt_status, "alert": False}
                    self.phone_near_face_start_time = None
                    self.phone_alert_active = False
                    self.phone_call_start_time = None
                    self.phone_call_active = False

                # Purely informational for the dashboard - no longer drives any alert.
                # Single source of truth: FaceStateMachine.absent_duration().
                no_face_duration = self.face_sm.absent_duration(now)

                # ==========================================================================
                # MODIFIED (Feature 5): if the face isn't present OR isn't stable yet,
                # EAR/MAR/pose/blink/fatigue computation stays frozen exactly as before
                # (no phantom detections off a missing/unstable face) - but face LOSS
                # itself now escalates through three realistic stages instead of doing
                # nothing until the driver comes back:
                #   0-2s  -> "Searching..."     (GREEN, no alarm - a blink, a quick
                #                                 glance down, or a single dropped
                #                                 frame is completely normal)
                #   2-5s  -> WARNING             (YELLOW, a single gentle voice nudge)
                #   5s+   -> DANGER               (RED, alarm - the driver genuinely
                #                                 isn't being monitored any more)
                # A face that's present but not yet "stable" (just reacquired) is
                # still treated as a neutral, alarm-free "STABILIZING..." state.
                # ==========================================================================
                if not is_tracking:
                    fatigue_score = self.fatigue_score_smoothed  # frozen, not recomputed
                    perclos, blink_rate = 0.0, 0.0
                    face_voice_message = None

                    # ---- MODIFIED (Phase 1): message/level chosen from the SINGLE
                    # face state instead of separate face_present/face_stable checks.
                    # SEARCHING and STABILIZING are neutral/GREEN (a blink, a quick
                    # glance down, or brief reacquisition is normal driving). FACE_LOST
                    # escalates YELLOW -> RED with elapsed absence duration. OCCLUDED
                    # is a placeholder here (the detector lands in a later Phase 1 fix)
                    # but the state/messaging path already exists end-to-end.
                    if self.face_sm.state == FaceState.SEARCHING:
                        level = "GREEN"
                        main_message = "SEARCHING FOR FACE..."
                        self.active_alerts.append("Searching for face...")

                    elif self.face_sm.state == FaceState.FACE_LOST:
                        if no_face_duration < self.cfg.FACE_LOST_WARNING_SECONDS:
                            level = "YELLOW"
                            main_message = "WARNING: FACE NOT VISIBLE"
                            self.active_alerts.append(f"WARNING: Face not visible ({no_face_duration:.1f}s)")
                            face_voice_message = "Please stay visible to the camera."
                        else:
                            level = "RED"
                            main_message = "DANGER: DRIVER NOT MONITORED"
                            self.active_alerts.append(f"DANGER: Face lost for {no_face_duration:.1f}s")
                            face_voice_message = "Driver monitoring lost. Please reposition."

                    elif self.face_sm.state == FaceState.OCCLUDED:
                        level = "GREEN"
                        main_message = "FACE PARTIALLY OCCLUDED"
                        self.active_alerts.append("Face partially occluded...")

                    else:  # STABILIZING
                        level = "GREEN"
                        main_message = "STABILIZING..."
                        self.active_alerts.append("Stabilizing face tracking...")

                    self.warning_level = level
                    self.update_alarm_and_voice(level, head_info, now, yawn_info, phone_info,
                                                 seatbelt_info, face_message=face_voice_message)

                else:
                    # ---------------- Alert aggregation ----------------
                    # PROBLEM 5: staged drowsiness (WARNING -> FATIGUE -> DROWSY ->
                    # CRITICAL) instead of one immediate "eyes closed = danger" cutoff.
                    # The audible alarm is gated on CRITICAL only (eyes_closed_alarm).
                    eye_stage = self.classify_eye_closure_stage(closed_duration) if eyes_closed else None
                    eyes_closed_alarm = eye_stage == "CRITICAL"
                    eyes_closed_warning = eye_stage in ("FATIGUE", "DROWSY", "CRITICAL")

                    if eye_stage == "CRITICAL":
                        self.active_alerts.append(f"CRITICAL DROWSINESS! Eyes closed {closed_duration:.1f}s")
                    elif eye_stage == "DROWSY":
                        self.active_alerts.append(f"DROWSY: Eyes closed {closed_duration:.1f}s")
                    elif eye_stage == "FATIGUE":
                        self.active_alerts.append(f"Fatigue: eye closure {closed_duration:.1f}s")
                    elif eye_stage == "WARNING":
                        self.active_alerts.append(f"Eye closure detected ({closed_duration:.1f}s)")

                    if head_info["instant_alert"] is not None:
                        self.active_alerts.append(f"INSTANT ALERT: HEAD {head_info['instant_alert']}")

                    if head_info["sudden_turn"]:
                        self.active_alerts.append("SUDDEN HEAD TURN")

                    for item in head_info["sustained_alerts"]:
                        # FIX (req #1): RED-level sustained head direction now uses
                        # the exact requested "HEAD DISTRACTION: {direction}" text;
                        # lower tiers (YELLOW/ORANGE) keep the existing progress text
                        # unchanged, since only the RED behavior was reported broken.
                        if item["level"] == "RED":
                            self.active_alerts.append(
                                f"HEAD DISTRACTION: {item['direction']} ({item['duration']:.1f}s)"
                            )
                        else:
                            self.active_alerts.append(
                                f"WARNING: LOOKING {item['direction']} ({item['duration']:.1f}s) [{item['level']}]"
                            )

                    if head_info["distracted"]:
                        self.active_alerts.append("DISTRACTED DRIVER")

                    if yawn_info.get("yawning"):
                        self.active_alerts.append(f"YAWN DETECTED (#{yawn_info['yawn_total']})")

                    if phone_info.get("phone_calling"):
                        self.active_alerts.append("PHONE CALLING DETECTED")
                    elif phone_info.get("phone_alert"):
                        self.active_alerts.append("PHONE DISTRACTION")

                    if seatbelt_info.get("alert"):
                        self.active_alerts.append("NO SEAT BELT")

                    # NEW (req #5): optional, throttled debug console diagnostics -
                    # console-only, never touches the dashboard.
                    self.print_debug_diagnostics(head_info, phone_info, yawn_info, now)

                    # ---------------- NEW (Issue 8): multi-signal drowsiness fusion ------
                    # A single threshold (EAR-closed-duration alone) is not required for
                    # everything: fuse several weaker, INDEPENDENT signals - eye closure,
                    # yawning, and abnormally low blink rate - so a combination of them
                    # can raise an early ORANGE warning before the full EYE_CLOSED_ALARM
                    # duration is reached, rather than doing nothing until one hard cutoff.
                    #
                    # FIX (false "drowsiness while eyes are open" bug): the EAR-below-
                    # threshold check and the closed-duration check used to be counted as
                    # TWO separate signals, but they are not independent evidence - by the
                    # time closed_duration reaches DROWSY_WARNING_SECONDS, avg_ear has
                    # necessarily been below threshold that whole time too. Counting both
                    # let a single real (but short/borderline) closure supply 2 of the 3
                    # "independent" signals needed, so any one weak extra signal (see next
                    # fix) tipped the fusion into a false alert even with eyes open most of
                    # the time. They are now merged into ONE "eye closure" signal.
                    drowsy_signal_count = 0
                    eye_weak_signal = (
                        (0.0 < avg_ear < self.dynamic_ear_threshold) or
                        (closed_duration >= self.cfg.DROWSY_WARNING_SECONDS)
                    )
                    if eye_weak_signal:
                        drowsy_signal_count += 1
                    if yawn_info.get("yawning") or yawn_info.get("yawn_just_confirmed"):
                        drowsy_signal_count += 1
                    # (blink-rate signal folded in below, once blink_rate is known)

                    fatigue_score, perclos, blink_rate = self.compute_fatigue_score(
                        now, head_info, yawn_info, phone_info
                    )
                    self.fatigue_score = fatigue_score

                    # FIX (false "drowsiness while eyes are open" bug, part 2): blink_rate
                    # is len(recent_blinks) / minutes_elapsed_in_window. In the first
                    # ~10-20s of ANY session (or right after calibration/reset), very few
                    # blinks have had time to happen yet no matter how alert the driver
                    # is, so blink_rate reads as artificially, permanently "low" - this
                    # signal used to fire almost every session regardless of real blink
                    # behavior. Only trust it once enough session time has passed for a
                    # blink RATE to be statistically meaningful.
                    elapsed_session = now - self.session_start_time
                    if (elapsed_session >= self.cfg.MIN_SESSION_SECONDS_FOR_BLINK_SIGNAL and
                            blink_rate < self.cfg.NORMAL_BLINKS_PER_MIN * 0.5):
                        drowsy_signal_count += 1

                    # Now only 3 truly independent signals are possible (eye closure,
                    # yawning, low blink rate over time) instead of the old 4 (with eye
                    # closure double-counted) - require 2-of-3 instead of 3-of-4 so a
                    # single momentary blink/glance can never reach it on its own, but a
                    # genuine early fatigue pattern (e.g. borderline eye closure PLUS
                    # yawning) still raises the early warning as originally intended.
                    fused_drowsy_warning = (not eyes_closed_alarm) and drowsy_signal_count >= 2
                    if fused_drowsy_warning:
                        self.active_alerts.append("FATIGUE PATTERN DETECTED (multiple signals)")

                    level = self.update_warning_level(
                        eyes_closed_alarm, head_info, fatigue_score,
                        phone_alert=phone_info.get("phone_alert", False),
                        seatbelt_alert=seatbelt_info.get("alert", False),
                        fused_drowsy_warning=fused_drowsy_warning,
                        eye_stage=eye_stage,
                        phone_calling=phone_info.get("phone_calling", False),
                        yawning_active=yawn_info.get("yawning", False),
                    )
                    self.warning_level = level

                    # ---------------- Alarm + voice trigger (feature 8) ----------------
                    self.update_alarm_and_voice(level, head_info, now, yawn_info, phone_info, seatbelt_info)

                    if level == "RED":
                        main_message = "DANGER: DRIVER NOT ALERT!"
                    elif level == "ORANGE":
                        main_message = "WARNING: STAY FOCUSED"
                    elif level == "YELLOW":
                        main_message = "CAUTION: STAY FOCUSED"
                    else:
                        main_message = "STATUS: ALL CLEAR"

                if self.calibrating and is_tracking:
                    main_message = "CALIBRATING..."

                # ---------------- Drawing ----------------
                self.draw_face_box(frame, bbox, level)

                info = {
                    "fps": self.fps,
                    # FIX (Phase 1, Fix #2): face_present/face_stable dict keys
                    # removed - draw_dashboard no longer reads them (it derives
                    # everything from face_state below, its single source of
                    # truth). Nothing else in the project reads them either
                    # (verified). is_tracking/landmarks_present remain available
                    # as local variables in run() for anything that still needs
                    # a plain boolean this frame.
                    "face_state": self.face_sm.state,
                    "avg_ear": avg_ear,
                    "left_ear": left_ear,
                    "right_ear": right_ear,
                    "eyes_closed": eyes_closed,
                    "closed_duration": closed_duration,
                    "blink_total": self.blink_total,
                    "blink_rate": blink_rate,
                    "pose": head_info.get("pose"),
                    "head_info": head_info,
                    "fatigue_score": fatigue_score,
                    "perclos": perclos,
                    "alerts": self.active_alerts,
                    "level": level,
                    "main_message": main_message,
                    "no_face_duration": no_face_duration,
                    "yawn_info": yawn_info,
                    "phone_info": phone_info,
                    "seatbelt_info": seatbelt_info,
                    "safety_score": self.compute_live_safety_score(now),
                    "confidence": self.face_sm.confidence(now, self._pose_confidence()),
                }
                # NEW (Part 4): keep a short rolling log of active-alert
                # reasons (with timestamps) for the "ALERT HISTORY" section of
                # the Trip Details panel - only appends when the leading
                # alert reason actually changes, so it isn't spammed every
                # single frame while one condition stays active.
                if self.active_alerts:
                    top_reason = self.active_alerts[0]
                    if not self.alert_history or self.alert_history[-1][1] != top_reason:
                        self.alert_history.append((now, top_reason))

                # ---------------- Compose the final console (Part 2D) ---------------
                # draw_dashboard() now returns a NEW canvas: a compact, resized
                # camera feed on the left + the large dashboard panel on the
                # right - it no longer mutates/overlays the raw camera frame.
                canvas = self.draw_dashboard(frame, info)

                # NEW (Feature 3): full-screen calibration overlay, now drawn on
                # top of the FULL composed canvas (camera + dashboard) so it
                # reads as one clean modal instead of only covering the camera.
                if self.calibrating:
                    self.draw_calibration_overlay(canvas, elapsed)

                cv2.imshow(window_name, canvas)

                # ---------------- Part 4: secondary "Trip Details" window -----------
                if self.show_details:
                    details_img = self.draw_details_panel(info)
                    cv2.imshow(self.details_window_name, details_img)
                    self._details_window_open = True
                elif self._details_window_open:
                    try:
                        cv2.destroyWindow(self.details_window_name)
                    except Exception:
                        pass
                    self._details_window_open = False

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
                elif key == ord('r'):
                    self.reset_session()
                elif key == ord('d'):
                    # Keyboard fallback for the TRIP DETAILS button (Part 4) -
                    # useful if a mouse isn't available/convenient.
                    self.show_details = not self.show_details

        finally:
            # ---------------- NEW (Feature 12): robust, leak-free resource cleanup ---
            # Every release step is independently guarded so a failure in one
            # (e.g. an already-closed camera, a mic/speaker driver hiccup) can
            # never prevent the others from running or crash the program on exit.
            try:
                self.alarm.stop()
            except Exception as exc:
                print(f"[Cleanup] Alarm stop failed: {exc}")
            try:
                if self.voice is not None:
                    self.voice.stop()
            except Exception as exc:
                print(f"[Cleanup] Voice manager stop failed: {exc}")
            try:
                if self.cap is not None:
                    self.cap.release()
            except Exception as exc:
                print(f"[Cleanup] Camera release failed: {exc}")
            try:
                self.face_mesh.close()
            except Exception as exc:
                print(f"[Cleanup] FaceMesh close failed: {exc}")
            try:
                cv2.destroyAllWindows()
            except Exception as exc:
                print(f"[Cleanup] Window cleanup failed: {exc}")
            # ---------------- Features 6/7: trip report + cloud sync -----------------
            self.generate_trip_report()


# =====================================================================================
# ENTRY POINT
# =====================================================================================
def main():
    print("=" * 70)
    print(" DRIVER MONITORING SYSTEM - Starting...")
    print(" Press 'q' or ESC to quit | Press 'r' to reset session stats")
    print("=" * 70)

    # FIX (req #8 - "program sometimes starts and immediately closes"):
    # DriverMonitoringSystem() construction itself used to be OUTSIDE any
    # try/except at all. Any exception raised during __init__ (e.g. the
    # MediaPipe/YOLO/camera setup this file does there) propagated straight
    # out uncaught - on Windows in particular, double-clicking a .py file (or
    # some IDE "run" configurations) closes the console the instant the
    # process exits, so the traceback flashes and vanishes before it can be
    # read, looking exactly like "starts and immediately closes" with no
    # visible reason. Every failure path below now (a) prints a clear,
    # specific message, (b) never hides the real exception, and (c) pauses
    # for a keypress before the process actually exits.
    dms = None
    exit_code = 0
    try:
        dms = DriverMonitoringSystem()
    except Exception as exc:
        exit_code = 1
        print(f"[ERROR] Initialization failed because: {exc}")
        import traceback
        traceback.print_exc()

    if dms is not None:
        try:
            dms.run()
        except RuntimeError as exc:
            exit_code = 1
            print(f"[ERROR] Main loop exited because: {exc}")
        except KeyboardInterrupt:
            print("\nInterrupted by user. Exiting.")
        except Exception as exc:
            # FIX (req #8): previously any non-RuntimeError exception here
            # (AttributeError, cv2.error, IndexError, etc.) was NOT caught at
            # all and would propagate out of main() uncaught - same silent-
            # close problem as above, just triggered mid-run instead of at
            # startup. Now every failure path is caught, explained, and
            # logged with a full traceback.
            exit_code = 1
            print(f"[ERROR] Main loop exited because: {exc}")
            import traceback
            traceback.print_exc()

    print("Driver Monitoring System closed.")
    # Keep the console window open long enough to actually read the message
    # above - critical on Windows, where the window can close instantly
    # otherwise. Skipped automatically in non-interactive environments.
    try:
        if sys.stdin is not None and sys.stdin.isatty():
            input("Press Enter to exit...")
    except Exception:
        pass
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
