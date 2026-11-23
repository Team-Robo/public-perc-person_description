#!/usr/bin/env python3
"""ROS2 node: visual appearance description for a tracked person, via VLM.

Serves /perception/people/describe. Input filtering (which frame to actually
hand the VLM) is the real design problem here, not the VLM call itself.

Matching is by NEAREST CAPTURE-TIME STAMP, not "whichever topic's cache was
most recently written to" - the same audit-F09 discipline perc-person_identity's
own fusion node already established for face<->body matching (see that
file's module docstring), extended here to also cover pose and the actual
image crop. A bounded, time-ordered history of recent BodyTrackArray/
PersonPoseArray/Image messages is kept; matching happens on FACE ARRIVAL
(the sparser, more selective signal - not every frame has a usable face),
searching each history for whichever entry's stamp is closest to the
face's own stamp, refused entirely if the closest candidate is still
farther than max_skew_sec away. Face<->body assignment uses head-region
association (ported from perc-person_identity's own _assoc_score - see
that file for why this beats whole-box IoU) rather than IoU on the two
full boxes.

The score combining everything is deliberately NOT just "face confidence
times shoulder confidence" - two more factors, motivated by real failure
modes found live in this repo's own testing:
  - sync_quality: how tightly the matched body/pose/image stamps actually
    cluster around the face's own stamp, decaying from 1.0 near-zero skew
    toward 0 at max_skew_sec. A tightly-synced candidate with decent (not
    perfect) face/shoulder scores should beat a loosely-synced one with
    marginally higher raw scores - more skew means more chance the person
    actually moved between signals.
  - size_quality: normalized crop width. A confidently-detected face and
    good shoulder score do NOT protect against a tiny, far-away crop -
    confirmed live: a real body_track_id's bbox was 96x352px on a
    1920x1080 frame (person far from camera, near the frame edge), which
    would have scored fine on face/shoulder alone but is genuinely too
    little detail for the VLM to read hair/clothing from.
"""

import logging
import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from person_body_interfaces.msg import BodyTrackArray
from person_description_interfaces.srv import DescribePerson
from person_face_interfaces.msg import FaceObservationArray
from person_pose_interfaces.msg import PersonPoseArray

# COCO-17 keypoint indices.
_LEFT_SHOULDER = 5
_RIGHT_SHOULDER = 6
_LEFT_HIP = 11
_RIGHT_HIP = 12

# Every /describe call's full input/output (raw VLM text, parsed attrs,
# composed sentence, per-stage timing) goes here instead of rosout/console -
# verbose and per-request, not the kind of thing you want scrolling past on
# every call when watching the dashboard. Append-mode, survives node
# restarts, so a debugging session's history isn't lost between test runs.
_IO_LOG_PATH = os.path.expanduser(
    os.environ.get('PERSON_DESCRIPTION_IO_LOG', '~/.ros/person_description_io.log')
)


def _make_io_logger() -> logging.Logger:
    logger = logging.getLogger('person_description.io')
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # never bubble up to rosout/console
    if not logger.handlers:  # guard against duplicate handlers on re-init
        os.makedirs(os.path.dirname(_IO_LOG_PATH), exist_ok=True)
        handler = logging.FileHandler(_IO_LOG_PATH, mode='a')
        handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
        logger.addHandler(handler)
    return logger


def _stamp_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _iou(a, b) -> float:
    """a, b: [x1, y1, x2, y2]."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _assoc_score(face_bbox, body_bbox, head_fraction: float) -> float:
    """How plausibly `face_bbox` belongs to `body_bbox`. Ported verbatim
    (logic, not code - separate repos) from perc-person_identity's own
    _assoc_score: 0 unless the face centre lies inside the body's own head
    band (top head_fraction of its box), otherwise 0.6 * (fraction of the
    face box contained in that band) + 0.4 * (how horizontally centred the
    face is in the body box). Head-region containment separates two people
    whose full bodies overlap far better than whole-box IoU, which has no
    notion of "does this face sit where a head should be on this body"."""
    fx = 0.5 * (face_bbox[0] + face_bbox[2])
    fy = 0.5 * (face_bbox[1] + face_bbox[3])
    bx1, by1, bx2, by2 = body_bbox
    bw = max(1e-6, bx2 - bx1)
    head_y2 = by1 + head_fraction * (by2 - by1)
    if not (bx1 <= fx <= bx2 and by1 <= fy <= head_y2):
        return 0.0
    ix1, iy1 = max(face_bbox[0], bx1), max(face_bbox[1], by1)
    ix2, iy2 = min(face_bbox[2], bx2), min(face_bbox[3], head_y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    farea = max(1e-6, (face_bbox[2] - face_bbox[0]) * (face_bbox[3] - face_bbox[1]))
    contain = min(1.0, inter / farea)
    centre = max(0.0, 1.0 - abs(fx - 0.5 * (bx1 + bx2)) / (0.5 * bw))
    return 0.6 * contain + 0.4 * centre


def _crop(img: np.ndarray, bbox) -> Optional[np.ndarray]:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2].copy()


def _match_pose(body_bbox, pose_people, pose_iou_min: float):
    """Whichever pose candidate (IoU >= pose_iou_min against body_bbox)
    gives the BEST shoulder score - returns (shoulder_score, keypoints)
    so a caller can build a torso crop from the SAME matched pose the
    shoulder gate already picked, instead of re-searching pose_people a
    second time with different selection logic."""
    best_score, best_keypoints = 0.0, None
    for pbbox, keypoints in pose_people:
        if _iou(body_bbox, pbbox) >= pose_iou_min:
            ls, rs = keypoints[_LEFT_SHOULDER], keypoints[_RIGHT_SHOULDER]
            s = (ls.score + rs.score) / 2.0
            if s > best_score:
                best_score, best_keypoints = s, keypoints
    return best_score, best_keypoints


def _torso_bbox(body_bbox, keypoints, torso_height_fraction: float, min_kp_score: float):
    """A tight torso-only region (shoulders to hips) built from pose
    keypoints, instead of clothing_color reading the WHOLE body_bbox -
    confirmed the body_track message is a head-to-feet box (see
    perc-person_body's BodyTrack.msg), so the old crop was mixing pants/
    skin/background pixels into what was supposed to be a shirt-color
    signal. Returns None if the shoulders aren't reliable enough to
    anchor a torso box at all - caller falls back to the full body crop.

    x-range is the shoulder-to-shoulder span widened 15% each side
    (clothing drapes past the shoulder points themselves - sleeves,
    looser shirts). y-range is real hip keypoints when both are
    confident, else a fixed fraction of body_bbox's own height below the
    shoulders (hips are frequently occluded/unreliable - seated, behind
    a table, cropped by frame edge - torso_height_fraction is an
    untuned typical shoulder-to-waist proportion, not measured live)."""
    ls, rs = keypoints[_LEFT_SHOULDER], keypoints[_RIGHT_SHOULDER]
    if ls.score < min_kp_score or rs.score < min_kp_score:
        return None
    bx1, by1, bx2, by2 = body_bbox
    shoulder_y = (ls.y + rs.y) / 2.0
    sx1, sx2 = min(ls.x, rs.x), max(ls.x, rs.x)
    width = max(1.0, sx2 - sx1)
    x1 = max(bx1, sx1 - 0.15 * width)
    x2 = min(bx2, sx2 + 0.15 * width)

    lh, rh = keypoints[_LEFT_HIP], keypoints[_RIGHT_HIP]
    if lh.score >= min_kp_score and rh.score >= min_kp_score:
        y2 = (lh.y + rh.y) / 2.0
    else:
        y2 = shoulder_y + torso_height_fraction * (by2 - by1)
    y1, y2 = shoulder_y, min(by2, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _expand_head_bbox(face_bbox, img_shape, top_frac: float, side_frac: float, bottom_frac: float):
    """InsightFace's own face bbox (RetinaFace-style, see perc-person_face's
    FaceObservation.msg) is a tight eyebrows-to-chin oval - fine for face
    recognition (all InsightFace itself needs), but hair_color/
    glasses_present/hat_present classification off THIS crop is mostly
    reading forehead/cheek skin, not hair, temples, or the top of the
    head where a hat actually sits. Expands asymmetrically before
    cropping - more headroom above (hair/hat) than sideways (temples,
    glasses arms) or below (a little chin/jaw/neck context) - clamped to
    the image bounds. Fractions are of the face box's OWN width/height,
    so the expansion scales with how close/far the face is, same as the
    box itself. Untuned starting points, not measured against real
    hair/hat coverage."""
    x1, y1, x2, y2 = face_bbox
    h, w = img_shape[:2]
    fw, fh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    ex1 = max(0.0, x1 - side_frac * fw)
    ex2 = min(float(w), x2 + side_frac * fw)
    ey1 = max(0.0, y1 - top_frac * fh)
    ey2 = min(float(h), y2 + bottom_frac * fh)
    return (ex1, ey1, ex2, ey2)


def _sharpness_score(img: np.ndarray) -> float:
    """Variance of the Laplacian - a standard, cheap blur proxy (both
    motion blur and defocus suppress high-frequency edge content, which
    is exactly what a Laplacian responds to). Near-zero for a flat/
    blurred image, growing with genuine edge detail. NOT bounded to
    [0,1] on its own - the caller normalizes against a target."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _nearest(buf: deque, target_ns: int):
    """Search a (stamp_ns, payload, frame_id) deque for the entry nearest
    target_ns. Returns (stamp_ns, payload, frame_id, abs_diff_ns), or None
    if buf is empty. Linear scan - same as perc-person_identity's own
    _faces_cb, fine at these buffer sizes (tens of entries)."""
    best = None
    for stamp_ns, payload, frame_id in buf:
        diff = abs(stamp_ns - target_ns)
        if best is None or diff < best[3]:
            best = (stamp_ns, payload, frame_id, diff)
    return best


@dataclass
class _BufferedFrame:
    image: np.ndarray        # body-bbox (head-to-feet) crop - the local/groq backends' only input, and debug topic
    head_image: np.ndarray   # face-bbox crop - hair_color/glasses_present/hat_present (clip_backend.py)
    torso_image: np.ndarray  # tight shoulders-to-hips crop (see _torso_bbox) - clothing_color (clip_backend.py)
    score: float
    stamp: float  # wall-clock time this candidate was accepted - for buffer_ttl_sec expiry only


class PersonDescriptionNode(Node):
    def __init__(self):
        super().__init__('person_description', namespace='perception')

        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('bodies_topic', '/perception/people/bodies')
        self.declare_parameter('faces_topic', '/perception/people/faces')
        self.declare_parameter('poses_topic', '/perception/people/poses')
        self.declare_parameter('head_fraction', 0.40)
        self.declare_parameter('min_assoc_score', 0.30)
        self.declare_parameter('pose_iou_min', 0.3)
        self.declare_parameter('min_shoulder_score', 0.3)
        self.declare_parameter('max_skew_sec', 0.15)
        self.declare_parameter('history_buffer_size', 90)
        self.declare_parameter('target_crop_width_px', 200.0)
        # Pure memory-hygiene safety net now, NOT a gate on whether
        # /describe can succeed - see _expire_buffer_locked's docstring
        # for why that used to be the same knob and why that was wrong.
        # Bumped way up from the old 5.0s: this only needs to be big
        # enough that a track nobody will ever query again eventually
        # gets garbage-collected, not tuned to "how long is a frame
        # still trustworthy" (that's staleness_halflife_sec below).
        self.declare_parameter('buffer_ttl_sec', 300.0)
        # Staleness now degrades GRACEFULLY (selection preference, via
        # an exponential half-life decay applied to a frame's score
        # before ensemble/MMR selection - see _on_describe) instead of a
        # hard cliff that deletes the whole buffer. Halves the effective
        # selection score of a non-exempt frame every this many seconds
        # of age - untuned starting point, not measured against how long
        # a guest's clothing/hair actually stays representative after
        # they look away.
        self.declare_parameter('staleness_halflife_sec', 20.0)
        # How many of the MOST RECENT frames (by wall-clock stamp, not
        # score) are fully exempt from the staleness penalty above - "the
        # actual k last" get to compete purely on quality, only frames
        # older than that get discounted for age. Default 1: the single
        # newest frame is never penalized, so a /describe call made the
        # instant a fresh frame lands never gets held back by staleness
        # math that has nothing to degrade yet.
        self.declare_parameter('recency_exempt_frames', 1)
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('load_model', True)
        # 'clip' (default): CLIP zero-shot classification (see
        # clip_backend.py) - local, small, structurally can't produce
        # malformed output OR confidently-wrong content from a
        # miscalibrated model, since there's no free-form generation
        # step at all, just cosine-similarity argmax over a fixed
        # candidate set. Live-tested and iterated extensively (crop
        # region fixes, MMR frame selection, mean/max ensemble pooling,
        # color-enum tuning) - promoted from experimental to default
        # after repeatedly outperforming both generative alternatives
        # below. 'local': Qwen2.5-VL-3B-Instruct, 4-bit quantized,
        # on-device (see vlm_backend.py) - confirmed live to produce both
        # malformed JSON and confidently-wrong attributes; kept as a
        # fallback/comparison option, not recommended. 'groq': Groq-
        # hosted qwen/qwen3.6-27b via API (see groq_vision_backend.py) -
        # traded 'local's accuracy problem for a hard network dependency;
        # same status as 'local', not recommended over 'clip'.
        self.declare_parameter('backend', 'clip')
        self.declare_parameter('groq_model_name', 'qwen/qwen3.6-27b')
        # How many of the buffered candidates CLIP actually classifies
        # over (mean-pool color, max-pool binary - see
        # clip_backend.py's describe_ensemble), chosen from
        # clip_candidate_pool via MMR (score + head-embedding diversity,
        # see _select_diverse) rather than plain top-k-by-score. local/groq
        # backends ignore every frame past the single best (frames[0]) -
        # harmless unused buffering for them, not worth branching out.
        self.declare_parameter('clip_ensemble_frames', 3)
        # How many recent top-scoring frames per body_track_id to keep
        # buffered - the CANDIDATE POOL that clip_ensemble_frames gets
        # selected FROM, deliberately larger than clip_ensemble_frames so
        # there's actually something to select among. Originally this WAS
        # the buffer cap directly (top-k-by-score, no selection) -
        # confirmed live that clusters on near-identical frames when a
        # person holds one pose for a second (same angle every frame),
        # which does nothing for an angle-dependent miss (glasses read
        # differently at different head angles) since averaging/max-
        # pooling over near-duplicates is barely different from one frame.
        # A wider pool gives the MMR selection actual angle diversity to
        # choose from instead of picking among near-clones.
        self.declare_parameter('clip_candidate_pool', 8)
        # MMR trade-off: 1.0 = pure top-k-by-score (no diversity benefit,
        # the original behavior), 0.0 = pure diversity (ignores score
        # entirely, could pick a blurry/occluded outlier just for being
        # different). 0.5 = equal weight - untested starting point, not
        # yet tuned against a real accuracy comparison.
        self.declare_parameter('clip_diversity_weight', 0.5)
        # Torso-crop construction (see _torso_bbox) - the clothing_color
        # crop used to be the whole body_bbox (head-to-feet, confirmed via
        # perc-person_body's BodyTrack.msg), mixing pants/skin/background
        # into the color read. torso_height_fraction is the shoulder-to-
        # waist fallback used when hip keypoints aren't reliable enough to
        # trust directly - an untuned typical human proportion, not
        # measured against this team's own camera/lens setup.
        self.declare_parameter('torso_height_fraction', 0.35)
        # Laplacian-variance blur/defocus penalty (see _sharpness_score) -
        # multiplied into the same frame-acceptance score as face/shoulder/
        # sync/size, added after live testing showed some of the highest-
        # scoring KEPT frames were visibly blurry despite scoring well on
        # every other factor (motion blur isn't detected by face
        # confidence, shoulder confidence, sync, or crop size at all).
        # Untuned - variance of the Laplacian is highly resolution/lens-
        # dependent, this default is a generic starting point, not a
        # value measured against this team's own camera.
        self.declare_parameter('target_sharpness_var', 150.0)
        # Head-crop expansion (see _expand_head_bbox) - InsightFace's own
        # face bbox is a tight eyebrows-to-chin oval with almost no hair/
        # hat region included, confirmed live to be a real problem for
        # hair_color/glasses_present/hat_present (clip_backend.py), all
        # three of which need to see hair, temples, or the top of the
        # head - none of which the raw face box reliably contains.
        # Fractions of the face box's own width/height; untuned.
        self.declare_parameter('head_crop_top_margin', 0.6)
        self.declare_parameter('head_crop_side_margin', 0.25)
        self.declare_parameter('head_crop_bottom_margin', 0.1)
        # Default False: buffering/debug-topic/service-server all start
        # immediately at zero VLM cost, and the model loads on the FIRST
        # real /describe call instead of blocking node startup. This is
        # what makes "watch the debug topic until a good frame shows up,
        # THEN call the service" a genuinely lightweight workflow (e.g.
        # driven from the dashboard's Graph tab + Services tree) rather
        # than needing the VLM resident the whole time just to buffer
        # frames. Set true to restore the old eager-at-startup behavior
        # (e.g. to front-load the one-time load latency before a live run).
        self.declare_parameter('eager_load', False)

        self._bridge = CvBridge()
        self._lock = threading.Lock()
        self._vlm_init_lock = threading.Lock()
        self._io_log = _make_io_logger()
        self.get_logger().info(f'per-request VLM input/output logged to {_IO_LOG_PATH}')

        history_size = int(self.get_parameter('history_buffer_size').value)
        self._body_buf: deque = deque(maxlen=history_size)   # (stamp_ns, {track_id: bbox}, frame_id)
        self._pose_buf: deque = deque(maxlen=history_size)   # (stamp_ns, [(bbox, keypoints), ...], frame_id)
        self._image_buf: deque = deque(maxlen=history_size)  # (stamp_ns, np.ndarray, frame_id)
        # Sorted best-first, capped at clip_candidate_pool - see that
        # param's own comment.
        self._best_frames: dict[int, list[_BufferedFrame]] = {}

        cb = ReentrantCallbackGroup()
        self.create_subscription(
            Image, self.get_parameter('image_topic').value,
            self._on_image, qos_profile_sensor_data, callback_group=cb)
        self.create_subscription(
            FaceObservationArray, self.get_parameter('faces_topic').value,
            self._on_faces, 10, callback_group=cb)
        self.create_subscription(
            PersonPoseArray, self.get_parameter('poses_topic').value,
            self._on_poses, 10, callback_group=cb)
        self.create_subscription(
            BodyTrackArray, self.get_parameter('bodies_topic').value,
            self._on_bodies, 10, callback_group=cb)

        # Debug only - republishes whichever crop most recently became the
        # new best-scoring frame for ANY tracked body, so this is watchable
        # live (dashboard Graph tab, rviz, foxglove) without needing to call
        # the service. /image matches perc-person_face/perc-person_body's
        # own /perception/debug/<node>/image convention (whole body_bbox
        # crop, unchanged); /head and /torso are ADDITIONAL - added so the
        # two crop-region fixes (_expand_head_bbox, _torso_bbox) are
        # actually visually checkable live instead of just trusting the
        # math, since both are untuned heuristics.
        self._debug_pub = self.create_publisher(
            Image, '/perception/debug/person_description/image', qos_profile_sensor_data)
        self._debug_head_pub = self.create_publisher(
            Image, '/perception/debug/person_description/head', qos_profile_sensor_data)
        self._debug_torso_pub = self.create_publisher(
            Image, '/perception/debug/person_description/torso', qos_profile_sensor_data)

        self._vlm = None
        backend = self.get_parameter('backend').value
        eager_load = self.get_parameter('eager_load').value
        if eager_load and backend == 'groq':
            # eager_load exists to front-load the LOCAL backend's multi-
            # second weight load ahead of a live run - the groq backend has
            # no weights to load (GroqVisionDescribeVLM.load_timing is
            # always 0), so eager_load:=true there is a no-op that only
            # misleads whoever set it into thinking something was
            # front-loaded. Not fatal - still just lazy-loads as normal.
            self.get_logger().warn(
                "eager_load:=true has no effect with backend:=groq (nothing "
                "to front-load) - don't set both; pick one")
        if not self.get_parameter('load_model').value:
            self.get_logger().warn('load_model:=false - service will return stub attrs, for wiring tests only')
        elif eager_load:
            self._ensure_vlm_loaded()
        else:
            self.get_logger().info(
                'VLM will lazy-load on the first real /describe call '
                '(eager_load:=true to load now instead)')

        self.create_service(
            DescribePerson, '/perception/people/describe', self._on_describe, callback_group=cb)
        self.get_logger().info('person_description ready: /perception/people/describe')

    def _ensure_vlm_loaded(self):
        """Idempotent, thread-safe - safe to call from every /describe request."""
        if self._vlm is not None:
            return
        with self._vlm_init_lock:
            if self._vlm is not None:  # a concurrent request may have won the race
                return
            backend = self.get_parameter('backend').value
            self.get_logger().info(f'loading VLM backend ({backend}, first call - can take a while)...')
            if backend == 'groq':
                # Importing this module runs its own load_dotenv() against
                # this repo's .env - GROQ_API_KEY is in os.environ by the
                # time the check below runs, if it's set anywhere at all.
                from person_description.groq_vision_backend import GroqVisionDescribeVLM
                if not os.environ.get('GROQ_API_KEY'):
                    self.get_logger().warn(
                        'GROQ_API_KEY not set - add it to '
                        'a .env file (see .env.example) or set GROQ_API_KEY / PERSON_DESCRIPTION_ENV_FILE '
                        '- requests will fail until then')
                vlm = GroqVisionDescribeVLM(model=self.get_parameter('groq_model_name').value)
            elif backend == 'clip':
                from person_description.clip_backend import ClipDescribeVLM
                vlm = ClipDescribeVLM(device=self.get_parameter('device').value)
            else:
                from person_description.vlm_backend import DescribeVLM
                vlm = DescribeVLM(device=self.get_parameter('device').value)
            load_timing_str = ' '.join(f'{k}={v * 1000:.0f}ms' for k, v in vlm.load_timing.items())
            self.get_logger().info(f'VLM backend ready ({load_timing_str})')
            self._vlm = vlm

    # --- buffer-append callbacks: no matching here, just record each
    # message's OWN capture stamp (never receipt time) into a bounded
    # history for _on_faces to search later. ---

    def _on_image(self, msg: Image):
        img = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        with self._lock:
            self._image_buf.append((_stamp_ns(msg.header.stamp), img, msg.header.frame_id))

    def _on_poses(self, msg: PersonPoseArray):
        people = [(tuple(p.bbox), p.keypoints) for p in msg.people]
        with self._lock:
            self._pose_buf.append((_stamp_ns(msg.header.stamp), people, msg.header.frame_id))

    def _on_bodies(self, msg: BodyTrackArray):
        boxes_by_id = {t.body_track_id: tuple(t.bbox) for t in msg.tracks}
        with self._lock:
            self._body_buf.append((_stamp_ns(msg.header.stamp), boxes_by_id, msg.header.frame_id))
            self._expire_buffer_locked(time.time())

    # --- the actual matching engine: triggered on face arrival, exactly
    # like perc-person_identity's own _faces_cb ---

    def _on_faces(self, msg: FaceObservationArray):
        if not msg.faces:
            return
        face_stamp_ns = _stamp_ns(msg.header.stamp)
        max_skew_ns = int(self.get_parameter('max_skew_sec').value * 1e9)

        with self._lock:
            body_match = _nearest(self._body_buf, face_stamp_ns)
            pose_match = _nearest(self._pose_buf, face_stamp_ns)
            image_match = _nearest(self._image_buf, face_stamp_ns)

        # Only an outright EMPTY buffer (nothing has arrived yet, e.g.
        # right at startup) hard-rejects here - a stale-but-present match
        # used to also be vetoed via `> max_skew_ns`, discarding the
        # candidate before sync_quality below ever got to weigh in at
        # all. Confirmed live that upstream body-track cadence isn't
        # always tight enough to land inside max_skew_ns, and this hard
        # veto was firing on most face events as a result - producing
        # constant warning spam AND zero usable frames, rather than just
        # a WORSE frame. sync_quality already decays continuously to 0.0
        # as skew grows past max_skew_ns (its `max(0.0, ...)` floor
        # handles arbitrarily large skew fine, not just skew near the
        # threshold) - that's the graceful degradation this should have
        # been relying on the whole time: a badly-synced candidate scores
        # near zero and gets outcompeted by anything better, instead of
        # being unconditionally thrown away and blocking the pipeline
        # from producing ANY frame when nothing better is available yet.
        if body_match is None:
            self.get_logger().warn(
                'no body-track snapshot buffered yet - skipping', throttle_duration_sec=2.0)
            return
        if image_match is None:
            self.get_logger().warn(
                'no image buffered yet - skipping', throttle_duration_sec=2.0)
            return
        # Pose is used for shoulder-confidence only, not required to even
        # attempt a match - missing/stale pose just means shoulder_score
        # stays 0 (min_shoulder_score gate below rejects the candidate).
        pose_ok = pose_match is not None and pose_match[3] <= max_skew_ns
        pose_people = pose_match[1] if pose_ok else []

        _, boxes_by_id, _, _ = body_match
        _, image, _, _ = image_match
        body_ids = list(boxes_by_id.keys())

        head_fraction = self.get_parameter('head_fraction').value
        min_assoc = self.get_parameter('min_assoc_score').value
        pose_iou_min = self.get_parameter('pose_iou_min').value
        min_shoulder = self.get_parameter('min_shoulder_score').value
        target_w = self.get_parameter('target_crop_width_px').value

        for face in msg.faces:
            best_body_id, best_assoc = None, 0.0
            for bid in body_ids:
                s = _assoc_score(tuple(face.bbox), boxes_by_id[bid], head_fraction)
                if s > best_assoc:
                    best_assoc, best_body_id = s, bid
            if best_body_id is None or best_assoc < min_assoc:
                continue

            body_bbox = boxes_by_id[best_body_id]
            shoulder_score, matched_keypoints = _match_pose(body_bbox, pose_people, pose_iou_min)
            if shoulder_score < min_shoulder:
                continue

            max_skew_observed_ns = max(
                body_match[3], image_match[3], pose_match[3] if pose_ok else 0)
            sync_quality = (max(0.0, 1.0 - max_skew_observed_ns / max_skew_ns)
                             if max_skew_ns > 0 else 1.0)

            crop_w = max(0.0, body_bbox[2] - body_bbox[0])
            size_quality = min(1.0, crop_w / target_w) if target_w > 0 else 1.0

            crop_img = _crop(image, body_bbox)
            head_bbox = _expand_head_bbox(
                tuple(face.bbox), image.shape,
                self.get_parameter('head_crop_top_margin').value,
                self.get_parameter('head_crop_side_margin').value,
                self.get_parameter('head_crop_bottom_margin').value)
            head_img = _crop(image, head_bbox)
            if crop_img is None or head_img is None:
                continue

            torso_img = None
            if matched_keypoints is not None:
                torso_height_fraction = self.get_parameter('torso_height_fraction').value
                tb = _torso_bbox(body_bbox, matched_keypoints, torso_height_fraction, min_shoulder)
                if tb is not None:
                    torso_img = _crop(image, tb)
            if torso_img is None:
                # No reliable keypoints to anchor a torso box - fall back
                # to the old full-body crop rather than dropping the
                # candidate entirely.
                torso_img = crop_img

            sharpness = _sharpness_score(crop_img)
            target_sharpness = self.get_parameter('target_sharpness_var').value
            sharpness_quality = min(1.0, sharpness / target_sharpness) if target_sharpness > 0 else 1.0

            score = (float(face.detection_confidence) * shoulder_score * sync_quality
                     * size_quality * sharpness_quality)

            new_frame = _BufferedFrame(
                image=crop_img, head_image=head_img, torso_image=torso_img, score=score, stamp=time.time())
            max_frames = max(1, int(self.get_parameter('clip_candidate_pool').value))
            with self._lock:
                frames = self._best_frames.setdefault(best_body_id, [])
                frames.append(new_frame)
                frames.sort(key=lambda f: f.score, reverse=True)
                del frames[max_frames:]
                # Identity check, not `in` - _BufferedFrame holds numpy
                # array fields, and dataclass equality on those raises
                # ("truth value of an array is ambiguous") rather than
                # comparing cleanly.
                accepted = any(f is new_frame for f in frames)
                if accepted:
                    # Always the current rank-0 frame, not necessarily the
                    # one just processed this call - this event could have
                    # entered the kept set at #2/#3 without displacing a
                    # still-better older frame at #1.
                    best = frames[0]
                    self._publish_debug_locked(self._debug_pub, best_body_id, best.image, best.score)
                    self._publish_debug_locked(self._debug_head_pub, best_body_id, best.head_image, best.score)
                    self._publish_debug_locked(self._debug_torso_pub, best_body_id, best.torso_image, best.score)
            if accepted:
                self.get_logger().info(
                    f'new frame kept for body_track_id={best_body_id}: score={score:.3f} '
                    f'(face={face.detection_confidence:.2f} shoulders={shoulder_score:.2f} '
                    f'sync={sync_quality:.2f} size={size_quality:.2f} sharp={sharpness_quality:.2f})')

    def _publish_debug_locked(self, publisher, body_track_id: int, crop: np.ndarray, score: float):
        msg = self._bridge.cv2_to_imgmsg(crop, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        # body_track_id/score aren't Image fields - carried in frame_id since
        # this is a debug-only topic (no typed carrier for "which track, how
        # good") rather than adding a dedicated message just for this.
        msg.header.frame_id = f'body_track_id={body_track_id} score={score:.3f}'
        publisher.publish(msg)

    def _expire_buffer_locked(self, now: float):
        """Pure memory hygiene, NOT a gate on whether /describe can
        succeed - that used to be the same knob, and it was wrong: this
        used to wipe a track's ENTIRE buffer the moment its newest frame
        crossed buffer_ttl_sec, so looking away from the robot for a few
        seconds made the next /describe call fail outright ("no
        describable frame buffered") even when several perfectly good
        frames from moments earlier were still sitting right there. A
        non-callable service is worse than an unconfident answer -
        describing someone from a slightly stale frame beats refusing to
        describe them at all. Staleness now only affects WHICH buffered
        frames get PREFERRED for the ensemble (see the halflife decay in
        _on_describe), continuously and with the most recent frame(s)
        exempt - never an outright buffer wipe. This function's only
        remaining job is bounding memory for tracks that will genuinely
        never be queried again, which is why buffer_ttl_sec is now a
        generous safety net rather than a tight recency requirement -
        and even past it, the single best-scoring frame is kept rather
        than dropping a track to zero."""
        ttl = self.get_parameter('buffer_ttl_sec').value
        for tid in list(self._best_frames.keys()):
            frames = self._best_frames[tid]
            fresh = [f for f in frames if f.stamp >= now - ttl]
            if not fresh and frames:
                fresh = [max(frames, key=lambda f: f.score)]
            self._best_frames[tid] = fresh

    # --- service ---

    def _on_describe(self, request, response):
        t_start = time.perf_counter()
        with self._lock:
            frames = self._best_frames.get(request.body_track_id)
        t_lookup = time.perf_counter()
        if not frames:
            response.success = False
            response.message = (
                f'no describable (front-facing) frame buffered for '
                f'body_track_id={request.body_track_id}')
            self.get_logger().info(
                f'describe: body_track_id={request.body_track_id} -> no frame buffered')
            self._io_log.info(
                f'body_track_id={request.body_track_id} buffer_lookup='
                f'{(t_lookup - t_start) * 1000:.1f}ms -> NO FRAME BUFFERED')
            return response

        try:
            if self.get_parameter('load_model').value:
                self._ensure_vlm_loaded()  # no-op after the first call; can raise (e.g. missing GROQ_API_KEY)
            if self._vlm and self.get_parameter('backend').value == 'clip':
                # Hand describe_ensemble the FULL candidate pool (up to
                # clip_candidate_pool) plus each frame's own score - it
                # runs MMR selection down to clip_ensemble_frames itself
                # (see describe_ensemble's own docstring), not just the
                # single best. Handles a list of any length, including 1.
                # torso_image (not image) - the tight shoulders-to-hips
                # crop, not the whole body_bbox (see _torso_bbox).
                #
                # Score fed to MMR is staleness-PENALIZED here, not the
                # raw quality score - a frame from 40 seconds ago should
                # lose out to a fresher merely-decent one when both are
                # in the pool, without ever being excluded outright (that
                # exclusion is exactly what _expire_buffer_locked used to
                # do wrong). The single most recent frame (by stamp, not
                # score) is exempt - "the actual k last" per
                # recency_exempt_frames - so a call made right as a fresh
                # frame lands is never held back by decay math with
                # nothing yet to decay.
                now = time.time()
                by_recency = sorted(frames, key=lambda f: f.stamp, reverse=True)
                exempt_n = max(0, int(self.get_parameter('recency_exempt_frames').value))
                exempt_ids = {id(f) for f in by_recency[:exempt_n]}
                halflife = float(self.get_parameter('staleness_halflife_sec').value)

                def _selection_score(f: _BufferedFrame) -> float:
                    if id(f) in exempt_ids or halflife <= 0:
                        return f.score
                    age = max(0.0, now - f.stamp)
                    return f.score * math.exp(-age / halflife)

                frame_pairs = [(f.head_image, f.torso_image, _selection_score(f)) for f in frames]
                attrs, vlm_timing, vlm_raw = self._vlm.describe_ensemble(
                    frame_pairs,
                    ensemble_k=int(self.get_parameter('clip_ensemble_frames').value),
                    diversity_weight=float(self.get_parameter('clip_diversity_weight').value),
                    return_timing=True)
            elif self._vlm:
                # local/groq only ever look at the single best frame -
                # generative one-shot backends, no ensembling story here.
                attrs, vlm_timing, vlm_raw = self._vlm.describe(frames[0].image, return_timing=True)
            else:
                attrs, vlm_timing, vlm_raw = _stub_attrs(), {}, ''
        except Exception as exc:  # noqa: BLE001 - report to caller, don't crash the node
            response.success = False
            response.message = f'inference failed: {exc}'
            self.get_logger().warn(f'describe: body_track_id={request.body_track_id} -> inference raised')
            self._io_log.info(
                f'body_track_id={request.body_track_id} buffer_lookup='
                f'{(t_lookup - t_start) * 1000:.1f}ms -> EXCEPTION: {exc!r}')
            return response
        t_done = time.perf_counter()

        threshold = self.get_parameter('confidence_threshold').value
        response.success = True
        response.hair_color = attrs['hair_color']
        response.hair_color_confidence = float(attrs['hair_color_confidence'])
        response.clothing_color = attrs['clothing_color']
        response.clothing_color_confidence = float(attrs['clothing_color_confidence'])
        response.glasses_present = _as_bool(attrs['glasses_present'])
        response.glasses_confidence = float(attrs['glasses_confidence'])
        response.hat_present = _as_bool(attrs['hat_present'])
        response.hat_confidence = float(attrs['hat_confidence'])
        response.description = _compose_sentence(attrs, threshold)
        response.message = 'ok'

        # rosout/dashboard console: one terse line per call.
        self.get_logger().info(
            f'describe: body_track_id={request.body_track_id} '
            f'total={(t_done - t_start) * 1000:.1f}ms -> "{response.description}"')

        # Everything else (raw VLM text, full per-attribute confidences,
        # per-stage timing) - the actual input/output record - goes to the
        # file only, so it's there to inspect but never scrolls past on
        # rosout/the dashboard console.
        timing_str = ' '.join(f'{k}={v * 1000:.1f}ms' for k, v in vlm_timing.items())
        attrs_str = (
            f"hair_color={attrs['hair_color']}({attrs['hair_color_confidence']:.2f}) "
            f"clothing_color={attrs['clothing_color']}({attrs['clothing_color_confidence']:.2f}) "
            f"glasses={attrs['glasses_present']}({attrs['glasses_confidence']:.2f}) "
            f"hat={attrs['hat_present']}({attrs['hat_confidence']:.2f})")
        self._io_log.info(
            f'body_track_id={request.body_track_id} '
            f'buffer_lookup={(t_lookup - t_start) * 1000:.1f}ms '
            f'vlm_total={(t_done - t_lookup) * 1000:.1f}ms ({timing_str}) '
            f'total={(t_done - t_start) * 1000:.1f}ms\n'
            f'  attrs: {attrs_str}\n'
            f'  description: "{response.description}"\n'
            f'  raw VLM output: {vlm_raw!r}')
        return response


def _stub_attrs() -> dict:
    return {
        'hair_color': 'unknown', 'hair_color_confidence': 0.0,
        'clothing_color': 'unknown', 'clothing_color_confidence': 0.0,
        'glasses_present': False, 'glasses_confidence': 0.0,
        'hat_present': False, 'hat_confidence': 0.0,
    }


def _as_bool(value) -> bool:
    """Safe bool coercion for a field that's SUPPOSED to already be a real
    bool but came from parsed model output. A model can emit a JSON-valid
    but wrongly-typed string like "false" for a boolean field - bool("false")
    is True in Python (any non-empty string is truthy), a real silent
    flip-to-wrong-answer bug this guards against."""
    if isinstance(value, str):
        return value.strip().lower() == 'true'
    return bool(value)


def _compose_sentence(attrs: dict, threshold: float) -> str:
    parts = []
    if attrs['hair_color_confidence'] >= threshold and attrs['hair_color'] != 'unknown':
        if attrs['hair_color'] == 'bald':
            parts.append('a bald head')
        else:
            parts.append(f"{attrs['hair_color']} hair")
    if attrs['clothing_color_confidence'] >= threshold and attrs['clothing_color'] != 'unknown':
        parts.append(f"a {attrs['clothing_color']} top")
    if attrs['glasses_confidence'] >= threshold and _as_bool(attrs['glasses_present']):
        parts.append('glasses')
    if attrs['hat_confidence'] >= threshold and _as_bool(attrs['hat_present']):
        parts.append('a hat')

    if not parts:
        return "I couldn't get a clear enough look to describe them."
    if len(parts) == 1:
        return f"They have {parts[0]}."
    return f"They have {', '.join(parts[:-1])}, and {parts[-1]}."


def main(args=None):
    rclpy.init(args=args)
    node = PersonDescriptionNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
