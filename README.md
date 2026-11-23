# perc-person_description

[![ROS 2 Humble](https://img.shields.io/badge/ROS_2-Humble-blue)](https://docs.ros.org/en/humble/) [![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-green.svg)](LICENSE) [![Maintainer](https://img.shields.io/badge/Maintainer-NTU_Team_Robo-orange)](mailto:teamrobontu@gmail.com)

> **Companion Packages:** Subscribes to `public-perc-person_body`, `public-perc-person_pose`, and `public-perc-person_face`. Python dependencies: [`requirements.txt`](requirements.txt).

Confidence-gated visual appearance description for tracked people on ROS 2 Humble — classifying
hair color, clothing color, glasses, and hat via **CLIP zero-shot classification** (default),
with a locally-hosted VLM (`Qwen2.5-VL-3B-Instruct`) or a Groq-hosted vision model available as
fallback/comparison options. Predictions are confidence-gated to suppress unreliable guesses rather
than forcing all four attributes on ambiguous crops.

**Why CLIP, not a VLM, is the default:** both generative VLM backends
were confirmed live to fail in ways CLIP is structurally incapable of -
`vlm_backend.py` (local Qwen2.5-VL) produced both malformed JSON and
confidently-wrong attributes; `groq_vision_backend.py` fixed the
malformed-output problem but kept the confidently-wrong-content one and
traded local VRAM cost for a hard network dependency. CLIP has no
free-form generation step at all (argmax over a fixed candidate set via
cosine similarity), so neither failure mode is possible by construction,
and extensive live iteration (crop-region fixes for both head and torso
crops, MMR frame selection for angle diversity, split mean/max ensemble
pooling, color-enum tuning) repeatedly out-performed both generative
alternatives across real test sessions. See `clip_backend.py`'s own
module docstring for the technical detail.

`hair_length` (short/medium/long) was replaced with `hat_present` because length is subjective
and unanswerable when headwear covers the hair, whereas `hat_present` is binary and visually
unambiguous like `glasses_present`.

---


## Architecture

```mermaid
flowchart LR
    Inputs["Bodies + Poses + Faces + RGB"] --> Node["person_description"]
    Node -->|"DescribePerson Srv"| Backend["CLIP / Qwen2.5-VL / Groq Backend"]
    Backend -->|"Confidence-Gated Attrs"| Client["HRI / Behavior Tree"]
```

## I. Requirements

- **OS / middleware:** Ubuntu 22.04, ROS 2 Humble.
- **Python (all backends):** dedicated venv `~/vlm-env` (own torch/
  transformers pins, same one-venv-per-stack pattern as `pose-env`/
  `reid-env`/`whisper-env` - don't install into any of those). CLIP and
  the local VLM backend share this same venv.
- **Model (default, `backend:=clip`):** `ViT-L-14` / `laion2b_s32b_b82k`,
  loaded via `open_clip` (same weights the old `laion/CLIP-ViT-L-14-
  laion2B-s32B-b82K` HF repo pointed at, confirmed live to produce the
  same classification decisions - the loading MECHANISM changed, not the
  model). ViT-L/14 specifically (not the smaller ViT-B/32) - confirmed
  live that patch32's coarser 7x7 spatial grid missed real, clearly-
  visible glasses that patch14's finer 16x16 grid catches. Small enough
  to run comfortably alongside the rest of the perception stack on the
  same GPU. `model_name`/`pretrained` are constructor args on
  `ClipDescribeVLM` - swapping the CLIP backbone is a two-string change,
  not new loading code (see `clip_backend.py`'s own module docstring).
  **MobileCLIP** (`model_name='MobileCLIP-S2'`,
  `pretrained='datacompdr'` - [arxiv.org/abs/2311.17049](https://arxiv.org/abs/2311.17049))
  tested live as an alternative: needs the extra `mobileclip` package
  (see II) for a required reparameterization step - skip it and
  MobileCLIP isn't just slower than advertised, it's SLOWER than the
  default ViT-L/14 (confirmed live: 66.6ms vs 65.9ms mean, std 44.5ms,
  one call hit 236ms) since its MobileOne-style conv blocks stay
  unfused. With reparameterization: a real 1.6x speedup (40.1ms mean,
  std 2.65ms) - `clip_backend.py` applies this automatically when
  `model_name` starts with `MobileCLIP`. Not switched to as the default
  though: on the same 3-photo test that caught the glasses/bald misses
  above, MobileCLIP-S2 missed real glasses at 55% confidence and
  confidently mis-called a man with visible (capped) hair "bald" at 67%
  - real accuracy regressions on exactly the cases the default model was
  already tuned to get right. Worth revisiting if inference latency ever
  becomes the actual bottleneck; not worth it for a same-or-better
  latency budget today.
- **Hardware:** an NVIDIA GPU with `device:=cuda` (falls back to CPU, not
  recommended - inference latency becomes the bottleneck for a live
  service call).
- **Fallback backend (`backend:=local`):** `Qwen/Qwen2.5-VL-3B-Instruct`,
  4-bit quantized (`bitsandbytes`) - confirmed live to produce both
  malformed JSON output and confidently-wrong attribute values (see
  `vlm_backend.py`'s own docstring); kept for comparison, not recommended
  over the default.
- **Fallback backend (`backend:=groq`):** no local GPU/model needed at
  all - offloads inference to a Groq-hosted `qwen/qwen3.6-27b` (vision-
  capable, `json_mode` support confirmed live against the account's own
  `/v1/models` endpoint). Needs `GROQ_API_KEY` in
  `perc-person_description/.env` (see `.env.example`, gitignored - never
  committed). Fixed `backend:=local`'s malformed-JSON problem but kept
  the confidently-wrong-content one, and trades local VRAM cost for a
  hard runtime dependency on internet + Groq's uptime/rate limits - same
  not-recommended status as `backend:=local`.

## II. Setup & Installation

`numpy<2` is required: newer NumPy removes the deprecated `numpy.Inf` alias that system `scipy`
(shared via ROS's Python path) still imports.

```bash
python3 -m venv ~/vlm-env
source ~/vlm-env/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
# Optional, only if testing model_name=MobileCLIP-*:
# pip install git+https://github.com/apple/ml-mobileclip.git --no-deps
```

Build the package in your ROS 2 workspace:

```bash
colcon build --packages-select person_description_interfaces person_description
source install/setup.bash
```

Note: on first run, `transformers` automatically downloads the
`Qwen2.5-VL-3B-Instruct` weights (~6 GB) to `~/.cache/huggingface`.

## III. How to Run

Start the `/perception/people/describe` server (default backend: CLIP).
**The model is not loaded yet at this point** - by default it lazy-loads
on the first real `/describe` call, so starting this node (and
`perc-person_pose` / `perc-person_face` alongside it) costs no GPU
beyond ordinary perception:

```bash
source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
PYTHONPATH=~/vlm-env/lib/python3.10/site-packages:$PYTHONPATH \
  ros2 run person_description person_description \
  --ros-args -p device:=cuda
```

**Recommended workflow (dashboard-driven, keeps the VLM off your GPU
until you actually want a description):**
1. Start `person_pose`, `person_face`, and this node together (e.g. as
   dashboard targets). No VLM cost yet - only detection/tracking runs.
2. Watch `/perception/debug/person_description/image` live in the
   dashboard's Graph tab (or rviz/foxglove) until a good frame shows up.
3. Use the dashboard's Services tree Call button (or `ros2 service call`)
   to fire `/perception/people/describe` with the `body_track_id` you saw
   in the debug image - *that* request is what triggers the one-time
   model load, then runs inference immediately after.

`eager_load:=true` restores the old startup-blocking behavior (load
immediately, before any request) - useful to front-load the one-time
load latency ahead of a live run instead of eating it on the first call:

```bash
ros2 run person_description person_description --ros-args -p eager_load:=true
```

`load_model:=false` disables the VLM entirely, always returning stub
attrs (all `"unknown"`/`0.0` confidence) - for pure wiring/graph tests:

```bash
ros2 run person_description person_description --ros-args -p load_model:=false
```

`backend:=local` uses the fallback local VLM (Qwen2.5-VL) instead of the
default CLIP backend - same `PYTHONPATH`, no extra setup, not
recommended over the default (see `I. Requirements`):

```bash
ros2 run person_description person_description --ros-args -p backend:=local
```

`backend:=groq` uses the Groq-hosted backend instead - no `PYTHONPATH`/GPU
needed, just `GROQ_API_KEY` in this repo's own `.env`. Don't combine with
`eager_load:=true` (warns at startup if you do) - there are no local
weights to front-load, so it's a no-op:

```bash
ros2 run person_description person_description --ros-args -p backend:=groq
```

Every `/describe` call's full input/output (raw model text, per-attribute
confidences, composed sentence, per-stage timing) is logged to
`~/.ros/person_description_io.log` (append mode) rather than
rosout/the dashboard console - verbose per-request detail that would
otherwise clutter both.

Request a description (`body_track_id` from `/perception/people/bodies`):

```bash
ros2 service call /perception/people/describe \
  person_description_interfaces/srv/DescribePerson "{body_track_id: 0}"
```

## IV. Repository Structure

```
perc-person_description/                    ← this repo (container, no package.xml of its own)
├── README.md
├── .env.example                            ← GROQ_API_KEY - copy to .env (gitignored) and fill in
├── person_description_interfaces/          ← DescribePerson.srv (ament_cmake)
│   ├── package.xml · CMakeLists.txt
│   └── srv/DescribePerson.srv
└── person_description/                     ← the node package
    ├── package.xml · setup.py · setup.cfg · resource/person_description
    ├── person_description/
    │   ├── person_description.py           ← the node: input filtering + /perception/people/describe
    │   ├── clip_backend.py                 ← CLIP zero-shot backend (backend:=clip, DEFAULT)
    │   ├── vlm_backend.py                  ← local Qwen2.5-VL backend (backend:=local, fallback)
    │   └── groq_vision_backend.py          ← Groq-hosted backend (backend:=groq, fallback), loads this repo's own .env
    └── test/
```

## V. Interfaces

What this node consumes and produces - the actual contract, same table
format as `perc-object_pointcloud`'s own README. Every topic name below
is a ROS parameter (its default), not hardcoded - see `III. How to Run`.

### Subscribed

| Topic (parameter) | Type | Notes |
|---|---|---|
| `/camera/color/image_raw` (`image_topic`) | `sensor_msgs/msg/Image` | Source image the actual crop is cut from. |
| `/perception/people/bodies` (`bodies_topic`) | `person_body_interfaces/msg/BodyTrackArray` | 2D track id + bbox per person - buffered, matched by capture-time stamp. |
| `/perception/people/faces` (`faces_topic`) | `person_face_interfaces/msg/FaceObservationArray` | Face detections + confidence. **This is the trigger** - matching runs on face arrival, the sparser signal (see `VI. Technical Explanation`). |
| `/perception/people/poses` (`poses_topic`) | `person_pose_interfaces/msg/PersonPoseArray` | 2D COCO-17 keypoints - shoulder confidence protects the clothing read specifically. |

### Published

| Topic | Type | Notes |
|---|---|---|
| `/perception/debug/person_description/image` | `sensor_msgs/msg/Image` | Debug only - whichever crop most recently became the new best-scoring frame for any tracked body. Watchable live (dashboard Graph tab, rviz, foxglove) without calling the service. |

### Services

| Service | Type | Notes |
|---|---|---|
| `/perception/people/describe` | `person_description_interfaces/srv/DescribePerson` | Request: `body_track_id` (int32). Response: `hair_color`/`clothing_color`/`glasses_present`/`hat_present` each with its own confidence, a composed `description` sentence (already confidence-gated), `success`/`message`. See `DescribePerson.srv` itself for the authoritative field list - it changes independently of this table. |

## VI. Technical Explanation

**How the default (CLIP) backend actually works.** Each of the four
attributes is a cosine-similarity argmax between an image embedding and
a small set of candidate text embeddings ("a photo of a person with
black hair" vs "...brown hair" etc.) - never free-form text generation,
so malformed output is structurally impossible. Two crops feed it, not
one: an expanded head crop (hair/glasses/hat - InsightFace's raw face
box is a tight eyebrows-to-chin oval with almost no hair/hat region,
expanded before use) and a tight torso crop derived from pose keypoints
(clothing color - NOT the whole body bbox, which was confirmed live to
mix in pants/skin/background pixels). When multiple recent frames are
buffered for the same person, `describe_ensemble` pools them
differently per attribute type: mean-pooled embeddings for hair/clothing
color (smooths per-frame lighting noise), max-pooled per-frame
classification for glasses/hat presence (a real feature can be
genuinely present but occluded in *some* frames without being absent -
averaging would wrongly punish that). Which frames make it into the
ensemble is chosen by MMR (Maximal Marginal Relevance: score + head-
embedding diversity), not just top-K-by-score, since near-identical
high-scoring frames from one angle don't help an angle-dependent
attribute the way genuinely different angles do. See `clip_backend.py`'s
own module/method docstrings for the full reasoning behind each of
these - this is a summary, not the authoritative source.

**Why a VLM (the fallback backends), not a trained attribute
classifier.** Closed-vocabulary PAR
(pedestrian attribute recognition) classifiers are trained on
surveillance-camera datasets (PA100K/PETA/RAP) - different angle,
distance, and lighting than a robot's close-range greeting-height camera.
That domain mismatch risks worse accuracy than a general-purpose VLM's
zero-shot read, on top of needing a labeled dataset and training pipeline
before deployment. A VLM needs neither.

**Input filtering is the actual design problem, not the VLM call - and
matching is by capture-time stamp, not "whichever cache was written to
most recently."** `/perception/people/bodies`, `/perception/people/faces`,
`/perception/people/poses`, and the raw image all publish independently,
at different rates, each with their own processing latency. Matching
"whatever's currently cached" against a live body message (no timestamp
comparison at all) can pair a body's current bbox against a stale face/
pose reading from a different moment - not hypothetical, this is the
exact failure mode `perc-person_identity`'s own fusion node was built to
fix for face<->body matching (see its module docstring, audit F09).
`person_description` follows the same discipline, extended to also cover
pose and the image crop itself:

- A bounded, time-ordered history of recent `BodyTrackArray`/
  `PersonPoseArray`/`Image` messages is kept, each keyed by its own
  `header.stamp` (never receipt time).
- Matching happens **on face arrival** - the sparser, more selective
  signal, mirroring `perc-person_identity`'s own design. For each
  incoming face, the body/pose/image histories are searched for whichever
  entry's stamp is nearest the face's own stamp; the whole event is
  refused if the closest candidate is still farther than `max_skew_sec`
  away.
- Face<->body assignment uses **head-region association** (containment in
  the top `head_fraction` of the body box + horizontal centering), ported
  from `perc-person_identity`'s own `_assoc_score` - this is what
  "confident face = front-facing" actually resolves to when there's more
  than one person in frame; whole-box IoU has no notion of "does this
  face sit where a head should sit on this body."
- **Shoulder confidence** (mean of the matched pose's left/right shoulder
  keypoint scores) protects the *clothing* read specifically - a good
  face shot with a motion-blurred or out-of-frame torso still gives a bad
  garment-color guess.

The score combining all of this is deliberately more than
`face_confidence x shoulder_confidence`:

```
score = face_confidence x shoulder_confidence x sync_quality x size_quality
```

- `sync_quality` - decays from 1.0 (near-zero skew) toward 0 as the
  worst skew among the matched body/pose/image stamps approaches
  `max_skew_sec`. A tightly-synced candidate with decent-but-not-perfect
  face/shoulder scores should beat a loosely-synced one with marginally
  higher raw scores - more skew means more chance the person actually
  moved between signals.
- `size_quality` - normalized crop width (`crop_width_px / target_crop_width_px`,
  capped at 1.0). A confidently-detected face and a good shoulder score
  do **not** protect against a tiny, far-away crop - confirmed live: a
  real `body_track_id`'s bbox was 96x352px on a 1920x1080 frame (person
  near the frame edge, far from camera), which would have scored fine on
  face/shoulder alone but is genuinely too little detail for the VLM to
  read hair/clothing from.

`person_description` keeps a rolling best-scoring frame per
`body_track_id`, buffered for `buffer_ttl_sec`, and the service just
returns whatever's currently buffered when asked.

**The service is confidence-gated, not just typed.** Each of the four
attributes carries its own confidence; `description` (the sentence handed
to `Speak`) only includes whichever attributes clear
`confidence_threshold` (default 0.5) - a stated-and-wrong attribute is
scored -20, not 0, so silence beats a guess.

## VII. References

- [CLIP-ViT-L-14-laion2B-s32B-b82K](https://huggingface.co/laion/CLIP-ViT-L-14-laion2B-s32B-b82K) -
  the default `backend:=clip` model (LAION-2B trained, safetensors).
- [CLIP (Radford et al. 2021)](https://arxiv.org/abs/2103.00020) -
  original paper, contrastive image-text pretraining.
- [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) - the
  fallback `backend:=local` model.
- [Groq API vision docs](https://console.groq.com/docs/vision) -
  `backend:=groq`'s `qwen/qwen3.6-27b`, `json_mode`, `reasoning_format`.
  Model id/availability drift over time - verify against
  `console.groq.com/docs/models` (or the account's own
  `/v1/models` endpoint, more authoritative than docs) before relying on
  it for a run, same caveat `hri-conversation_llm`'s own `GroqProvider`
  notes for its model.
- `Team-Robo/public-perc-person_identity` - source of the capture-time-stamp
  fusion discipline (audit F09) and the head-region association pattern
  (`_assoc_score`) this repo ports.

---

## Contributors

- **Khiem Nguyen** (`@KhiemGOM`)

Maintained by **Nanyang Technological University Team Robo** (`teamrobontu@gmail.com`).
## License

Apache-2.0 — see [LICENSE](LICENSE).
