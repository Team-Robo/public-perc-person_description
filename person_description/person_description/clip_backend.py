"""CLIP zero-shot classifier: two image crops (head, torso) -> structured
appearance attributes, via cosine-similarity ranking against a small fixed
set of candidate text labels per attribute - never free-form text
generation. Real teams use this pattern for exactly this task: CATIE
Robotics' RoboCup@Home TDP references training/matching image-text pairs
describing hair color/clothes/accessories, and RoBorregos' home-vision
uses a dedicated small model (not a generative VLM) for shirt color.

Structurally can't hit either failure mode the two generative backends
(vlm_backend.py, groq_vision_backend.py) were confirmed live to hit:
malformed output (there's no generation step - argmax over a fixed,
always-valid candidate set) or confidently-wrong content from a
miscalibrated model (softmax over cosine similarities is a much better-
behaved confidence signal than a generative model's self-reported one).

Two crops, not one, because the four attributes don't share a region:
hair_color/glasses_present/hat_present need a head/face crop,
clothing_color needs the torso. person_description.py computes both
(.head_image is an EXPANDED face-bbox crop - InsightFace's own face box
is a tight eyebrows-to-chin oval with almost no hair/hat region
included, confirmed live to be a real problem for exactly the three
attributes that need to see hair/temples/top-of-head; .torso_image is a
tight shoulders-to-hips crop derived from pose keypoints, NOT the whole
body_bbox - that used to feed clothing_color directly and confirmed
live to mix in pants/skin/background pixels, part of why color reads
were noisy). Both crop fixes live in person_description.py, not here -
this file only ever sees the crop it's handed.

clothing_color briefly moved to classical dominant-color matching
(pixel median + Lab-distance against a fixed color table) - reverted
back to CLIP text-similarity here since the torso-crop fix alone was
the higher-value change to isolate and verify first; the dominant-color
approach is still worth revisiting later (git history has it) if the
crop fix alone doesn't fully resolve the color-read noise.

Same describe(..., return_timing=True) -> (attrs, timing, raw) contract as
the other two backends, EXCEPT this one takes two crops
(describe(head_crop, torso_crop, ...)) since it genuinely needs both -
person_description.py branches on which call shape to use per backend.
"raw" here is a formatted dump of every candidate's similarity score per
attribute, not a raw text blob - there's no generation to show, but the
per-candidate scores are exactly the debug signal you'd want instead.
"""

import time

import numpy as np
import torch
from PIL import Image as PILImage

# openai/clip-vit-base-patch32 has no safetensors weights (only the old
# pickle-based pytorch_model.bin) - transformers hard-blocks torch.load on
# pickle checkpoints below torch 2.6 (CVE-2025-32434, a real vulnerability
# fix, not a bug to route around), and this venv's torch (2.5.1, shared
# with the local Qwen2.5-VL backend and bitsandbytes - not safe to bump
# unilaterally here) predates that. LAION-2B ships proper safetensors,
# sidestepping torch.load entirely.
#
# ViT-B/32 (patch32) was confirmed LIVE to miss real, clearly-visible
# glasses at 0.77 confidence - not a borderline miss like a fine color
# distinction, a confident wrong answer on the textbook CLIP zero-shot
# use case (binary presence/absence). Root cause candidate: patch32 means
# a 224x224 input is only a 7x7 grid of 32x32-pixel patches - very coarse
# spatially, easily losing thin details like glasses temple arms. Switched
# to ViT-L/14 (14x14 patches - a 16x16 grid, ~5x finer spatially - and a
# larger ~428M-param backbone vs ViT-B/32's ~150M) - still dramatically
# smaller than either VLM backend (3B/27B), and finer patches should
# preserve small accessory details ViT-B/32 was confirmed to lose.
#
# Loaded via open_clip, not transformers.CLIPModel/CLIPProcessor (what
# this file used before) - switched so a DIFFERENT checkpoint family
# (MobileCLIP: arxiv.org/abs/2311.17049, Apple's distilled/efficient CLIP
# variant, claimed 2.3x faster than a same-tier ViT-B/16 CLIP while more
# accurate) can be tried through the exact same loading code, since
# MobileCLIP ships through open_clip's registry, not as a transformers-
# compatible checkpoint. open_clip is also where LAION's own checkpoints
# actually come from - this model_name/pretrained pair
# ('ViT-L-14'/'laion2b_s32b_b82k') is confirmed to resolve to the exact
# same weights the old laion/CLIP-ViT-L-14-laion2B-s32B-b82K HF repo
# pointed at, so this refactor changes the LOADING MECHANISM only, not
# the model - default behavior is unchanged. Swapping models (e.g. to try
# MobileCLIP-S2/'datacompdr') is a two-string constructor argument change
# now, not new loading code.
_DEFAULT_MODEL_NAME = 'ViT-L-14'
_DEFAULT_PRETRAINED = 'laion2b_s32b_b82k'

# CLIP was trained on "a photo of X"-style web captions - phrasing
# candidates this way matches its training distribution far better than
# bare words ("black" vs "a photo of a person with black hair").
#
# Widened after live testing showed a forced-choice symptom: confidence
# split near-evenly between two ADJACENT candidates across repeated
# calls for what should be one stable real hair color (black/brown
# ~0.48/0.48 on one call). The softmax has to pick something from the
# fixed candidate set even when the true color sits BETWEEN two listed
# options - dark_brown/white give those in-between real-world colors
# their own bucket instead of splitting votes with their nearest
# neighbor. clothing_color had the exact same symptom (brown/red trading
# the top spot call after call on what's presumably one shirt) - kept
# widened here too, on the theory that most of that noise was actually
# the CROP being wrong (whole body_bbox, not the shirt region - see
# person_description.py's _torso_bbox) rather than the label set itself.
_HAIR_COLOR_LABELS = {
    'a photo of a person with black hair': 'black',
    'a photo of a person with dark brown hair': 'dark brown',
    'a photo of a person with brown hair': 'brown',
    'a photo of a person with blonde hair': 'blonde',
    'a photo of a person with gray hair': 'gray',
    'a photo of a person with white hair': 'white',
    'a photo of a person with red hair': 'red',
}
_CLOTHING_COLORS = [
    'black', 'white', 'blue', 'red', 'green', 'yellow', 'gray', 'brown', 'orange', 'purple', 'pink',
    'maroon', 'navy', 'beige', 'olive', 'teal',
]
_CLOTHING_COLOR_LABELS = {f'a photo of a person wearing a {c} shirt': c for c in _CLOTHING_COLORS}
_GLASSES_LABELS = {
    'a close-up photo of a person wearing glasses': True,
    'a close-up photo of a person not wearing glasses': False,
}
# hair_length was dropped workspace-wide (see DescribePerson.srv) - too
# subjective a judgment call, and unanswerable when a hat's on anyway.
# hat_present is binary, same reasoning as glasses_present for using the
# head crop - but NOT equally visually obvious to CLIP in practice.
# Confirmed live: the bare word "hat" confidently (0.87) called a man
# wearing an unmistakable gray flat cap "no hat," and separately scored
# an ornate Edwardian hat in a period B&W photo as a coin flip (0.48).
# Tested 5 phrasings against both failures plus a real no-hat case before
# picking this one - "hat or cap"/"headwear" alone each fixed only ONE of
# the two failures, "cap only" fixed both but weakened the no-hat case
# more; explicitly listing hat+cap+headwear together and contrasting
# against "bare head" (not just "not wearing X") is what actually fixed
# BOTH misses (cap photo 0.13->0.998, Edwardian hat 0.48->0.987) while
# staying correctly negative on the bald/no-hat control (0.83->0.66 -
# still right, just less overconfident). glasses_present did NOT have
# this problem when tested the same way - widening it to "eyewear"/
# "spectacles" only made its one real positive case LESS confident
# (0.836->0.56-0.75), so that one is left as the bare word.
_HAT_LABELS = {
    'a close-up photo of a person wearing a hat, cap, or other headwear': True,
    'a close-up photo of a person with a bare head, not wearing any headwear': False,
}
# A bald subject used to have no correct answer available at all - forced
# to pick from 7 actual hair colors with no "there is no hair" option,
# confirmed live to confidently call a bald scalp "red hair" (0.79). Fixed
# as its own binary gate ahead of the color question, same shape as
# glasses/hat above, NOT as an 8th color competing in the same softmax -
# baldness isn't a point in color space, so it has no Lab swatch and never
# enters the correlation kernel below. Color classification only runs at
# all when this comes back False.
_BALD_LABELS = {
    'a close-up photo of a bald person with no hair': True,
    'a close-up photo of a person with visible hair on their head': False,
}

# --- Correlated-neighbor confidence boost for hair_color/clothing_color ---
#
# First version of this (git history) redistributed probability mass by
# convolving the raw softmax with a label-similarity kernel (smoothed = K @
# raw). Confirmed LIVE that this is the wrong shape of fix: on a real photo
# where CLIP's own raw read was black=66%/dark_brown=14%/brown=15%, that
# redistribution PULLED black's confidence DOWN to 44% - correlated support
# from real neighbors made the top pick look LESS certain, backwards from
# what correlated support should mean. Redistribution can't avoid this: any
# scheme that gives every label a symmetric cut of its neighbors' mass
# necessarily takes some share away from whoever's currently winning, no
# matter how the kernel is weighted.
#
# What we actually want (confirmed against the same real numbers before
# writing this): when the top label has real, correlated neighbor support,
# that's CORROBORATING evidence and should raise its confidence, not spread
# it thinner. So only the current argmax gets a bonus - built from its own
# neighbors' raw scores - and nothing else moves:
#     boosted[top] = raw[top] + BETA * sum(K[top, j] * raw[j] for j != top)
#     boosted[i]   = raw[i]  for every other label i
#     confidence   = boosted / sum(boosted)
# BETA=0 recovers plain CLIP softmax exactly. On the same real case this
# takes black 66%->71% at BETA=2 (dark_brown/brown corroborating, real
# neighbors at dE 20-22) while a synthetic isolated 80%-confidence spike
# with no real neighbors nearby moves less than 1 point at BETA=2 - the
# boost only fires when there's genuine nearby support to draw on, it
# doesn't manufacture confidence out of nothing.
#
# K itself (the neighbor-closeness definition the boost draws on) is
# unchanged from the redistribution version: built from CIE Lab distance
# (CIE76 dE) between a canonical RGB swatch per label, NOT from CLIP's own
# text-embedding cosine distance - using CLIP's own geometry would be
# circular, since a systematic CLIP bias between two colors would just get
# laundered into a more confident-looking wrong answer instead of being
# caught by it. Lab swatch distance is an independent source of truth.
#
# Swatches are standard CSS3 named-color hex values for clothing (a shirt's
# named color maps reasonably well onto the CSS palette) and hand-picked
# representative tones for hair (hair "red"/"brown"/etc. read differently
# than fabric of the same name - a redhead is closer to auburn than to CSS
# pure red #FF0000). Both are reference approximations, not measured
# against real photos - worth revisiting if live testing shows a specific
# pair boosting the wrong way.
#
# Sigma=18 chosen from the actual computed CIE76 distances between these
# specific swatches: adjacent/confusable bins sit at dE~13-22 (maroon/brown
# 14.2, beige/white 13.4, black/dark_brown 22.1, dark_brown/brown 20.3,
# gray/white 22.4) while genuinely different colors sit at dE 40-180
# (maroon/red 51.4 - these read as different enough NOT to correlate,
# despite both being "reddish" by name; red/blue 176.3; teal/blue 147.7).
_COLOR_KERNEL_SIGMA = 18.0
# How much weight neighbor corroboration gets relative to the top label's
# own raw score. Picked by checking the real logged case above across a
# range (beta=1: 66.4%->68.8%, beta=2: ->70.8%, beta=4: ->74.3%,
# beta=8: ->79.2%) - 2.0 gives a real, visible boost without running away
# to false-certainty on one data point.
_COLOR_BOOST_BETA = 2.0

_CLOTHING_COLOR_SWATCHES_HEX = {
    'black': '#000000', 'white': '#FFFFFF', 'blue': '#0000FF', 'red': '#FF0000',
    'green': '#008000', 'yellow': '#FFFF00', 'gray': '#808080', 'brown': '#A52A2A',
    'orange': '#FFA500', 'purple': '#800080', 'pink': '#FFC0CB', 'maroon': '#800000',
    'navy': '#000080', 'beige': '#F5F5DC', 'olive': '#808000', 'teal': '#008080',
}
_HAIR_COLOR_SWATCHES_HEX = {
    'black': '#0B0B0B', 'dark brown': '#3B2314', 'brown': '#6F4E37', 'blonde': '#E8C39E',
    'gray': '#B0B0B0', 'white': '#F0EEE9', 'red': '#A54A2A',
}


def _hex_to_rgb(hex_str: str) -> np.ndarray:
    h = hex_str.lstrip('#')
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float64)


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = c / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Standard sRGB (D65) -> CIE Lab, vectorized over a leading batch dim."""
    lin = _srgb_to_linear(rgb)
    r, g, b = lin[..., 0], lin[..., 1], lin[..., 2]
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
    xn, yn, zn = 0.95047, 1.00000, 1.08883
    x, y, z = x / xn, y / yn, z / zn
    delta = 6.0 / 29.0
    f = lambda t: np.where(t > delta ** 3, np.cbrt(t), t / (3 * delta ** 2) + 4.0 / 29.0)
    fx, fy, fz = f(x), f(y), f(z)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    bb = 200 * (fy - fz)
    return np.stack([L, a, bb], axis=-1)


def _build_color_kernel(labels: list, swatches_hex: dict, sigma: float) -> np.ndarray:
    """labels order must match the per-label score vectors this kernel is
    applied to. Returns K[i,j] = exp(-dE76(i,j)^2 / (2*sigma^2)), K[i,i]=1."""
    labs = np.stack([_rgb_to_lab(_hex_to_rgb(swatches_hex[label])) for label in labels])
    diffs = labs[:, None, :] - labs[None, :, :]
    delta_e = np.sqrt((diffs ** 2).sum(-1))
    return np.exp(-(delta_e ** 2) / (2 * sigma ** 2))


class ClipDescribeVLM:
    def __init__(
            self, device: str = 'cuda', model_name: str = _DEFAULT_MODEL_NAME,
            pretrained: str = _DEFAULT_PRETRAINED):
        t0 = time.perf_counter()
        # Imported lazily, same reasoning as the other two backends - the
        # ROS node stays importable without torch on the path unless this
        # backend is actually selected.
        import open_clip
        t1 = time.perf_counter()

        self._device = device
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device)
        self._model.eval()
        if model_name.startswith('MobileCLIP'):
            # MobileCLIP's conv blocks are MobileOne-style: trained as
            # several parallel branches, mathematically fusible into one
            # branch for inference (same weights, same predictions -
            # confirmed live: identical attrs before/after on a real
            # photo). Skipping this is a real trap, not a micro-
            # optimization to skip if lazy - confirmed live it left
            # MobileCLIP-S2 not just slower than advertised but SLOWER
            # than our current ViT-L/14 on describe() (66.6ms vs 65.9ms,
            # no win) with wild run-to-run variance (std 44.5ms, one call
            # hit 236ms) - each un-fused branch is its own CUDA kernel,
            # and the mix apparently doesn't cache/schedule predictably.
            # Reparameterized: 40.1ms mean, std 2.65ms - the real 1.6x
            # speedup the model is actually capable of. Guarded to this
            # model family specifically (not called on the default ViT-
            # L/14 path) since reparameterize_model expects MobileOne-
            # style modules that a plain ViT doesn't have.
            from mobileclip.modules.common.mobileone import reparameterize_model
            self._model = reparameterize_model(self._model)
            self._model.to(device).eval()
        self._tokenizer = open_clip.get_tokenizer(model_name)
        t2 = time.perf_counter()

        # Text embeddings are static (don't depend on the image) - computed
        # ONCE here, not per describe() call, so a describe() call only
        # ever pays for the image side.
        self._hair_color_labels = list(_HAIR_COLOR_LABELS.values())
        self._clothing_color_labels = list(_CLOTHING_COLOR_LABELS.values())
        self._glasses_labels = list(_GLASSES_LABELS.values())
        self._hat_labels = list(_HAT_LABELS.values())
        self._bald_labels = list(_BALD_LABELS.values())
        self._hair_color_emb = self._embed_texts(list(_HAIR_COLOR_LABELS.keys()))
        self._clothing_color_emb = self._embed_texts(list(_CLOTHING_COLOR_LABELS.keys()))
        self._glasses_emb = self._embed_texts(list(_GLASSES_LABELS.keys()))
        self._hat_emb = self._embed_texts(list(_HAT_LABELS.keys()))
        self._bald_emb = self._embed_texts(list(_BALD_LABELS.keys()))

        # Label-similarity kernels for correlated-neighbor smoothing (see the
        # big comment above _COLOR_KERNEL_SIGMA) - only hair_color/
        # clothing_color need this; glasses/hat are already a forced 2-way
        # choice with no third neighbor to borrow mass from or lend it to.
        self._hair_color_kernel = _build_color_kernel(
            self._hair_color_labels, _HAIR_COLOR_SWATCHES_HEX, _COLOR_KERNEL_SIGMA)
        self._clothing_color_kernel = _build_color_kernel(
            self._clothing_color_labels, _CLOTHING_COLOR_SWATCHES_HEX, _COLOR_KERNEL_SIGMA)
        t4 = time.perf_counter()

        self.load_timing = {
            'import': t1 - t0, 'model_and_transforms': t2 - t1,
            'text_embeddings': t4 - t2, 'total': t4 - t0,
        }

    def _embed_texts(self, texts: list) -> torch.Tensor:
        tokens = self._tokenizer(texts).to(self._device)
        with torch.no_grad():
            emb = self._model.encode_text(tokens)
        return emb / emb.norm(dim=-1, keepdim=True)

    def _embed_image(self, bgr_crop: np.ndarray) -> torch.Tensor:
        rgb = bgr_crop[:, :, ::-1]
        pil_img = PILImage.fromarray(rgb)
        tensor = self._preprocess(pil_img).unsqueeze(0).to(self._device)
        with torch.no_grad():
            emb = self._model.encode_image(tensor)
        return emb / emb.norm(dim=-1, keepdim=True)

    def _classify(self, image_emb: torch.Tensor, text_emb: torch.Tensor, labels: list):
        # logit_scale-weighted cosine similarity through softmax - CLIP's
        # own convention (same scale it was trained with) - turns raw
        # similarities into a genuine probability distribution over the
        # candidate set, not just a ranking.
        logit_scale = self._model.logit_scale.exp()
        sims = (logit_scale * image_emb @ text_emb.T).softmax(dim=-1)[0]
        best_idx = int(sims.argmax())
        per_label = {labels[i]: float(sims[i]) for i in range(len(labels))}
        return labels[best_idx], float(sims[best_idx]), per_label

    def _classify_smoothed(
            self, image_emb: torch.Tensor, text_emb: torch.Tensor, labels: list, kernel: np.ndarray,
            beta: float = _COLOR_BOOST_BETA):
        """Same CLIP softmax as _classify, then an asymmetric confidence
        boost applied ONLY to the raw argmax (see the big comment above
        _COLOR_KERNEL_SIGMA for why symmetric redistribution was wrong).
        argmax is picked on the RAW scores first - the boost can make the
        winner more confident, it never changes who wins, since every
        other label's score is untouched and only compared against the
        one boosted value. Returns raw AND boosted per-label dicts (the
        gap between them on the top label IS the corroboration signal)."""
        logit_scale = self._model.logit_scale.exp()
        sims = (logit_scale * image_emb @ text_emb.T).softmax(dim=-1)[0]
        raw = np.array([float(sims[i]) for i in range(len(labels))])
        top = int(np.argmax(raw))
        neighbor_support = float(kernel[top] @ raw - raw[top])  # excludes self (kernel diagonal is 1)
        boosted = raw.copy()
        boosted[top] = raw[top] + beta * neighbor_support
        boosted = boosted / boosted.sum()
        per_label_raw = {labels[i]: float(raw[i]) for i in range(len(labels))}
        per_label_boosted = {labels[i]: float(boosted[i]) for i in range(len(labels))}
        return labels[top], float(boosted[top]), per_label_boosted, per_label_raw

    @staticmethod
    def _average_embeddings(embs: list) -> torch.Tensor:
        """Mean of several L2-normalized embeddings, re-normalized - the
        standard way to combine multiple noisy observations of the same
        subject in CLIP's embedding space. Each individual embedding
        already lives on the unit hypersphere; averaging then
        renormalizing keeps the combined vector on it too, rather than
        classifying each observation independently and voting on
        already-decided labels (which would only smooth out disagreements
        AFTER they've already hardened into a specific wrong label).

        Used for hair_color/clothing_color specifically - a wrong color
        read isn't announced by low confidence (bad lighting can be just
        as confident as good lighting), so averaging the raw signal
        before classifying is what actually smooths that out. NOT used
        for glasses_present/hat_present - see _classify_max_pool for why
        those need the opposite treatment."""
        stacked = torch.cat(embs, dim=0)  # (N, D)
        mean = stacked.mean(dim=0, keepdim=True)  # (1, D)
        return mean / mean.norm(dim=-1, keepdim=True)

    def _classify_max_pool(self, image_embs: list, text_emb: torch.Tensor, labels: list):
        """Multiple-instance-learning-style max pooling: classify each
        frame's OWN embedding independently, then let each candidate
        label be scored by its single best-supporting frame, not an
        averaged embedding. For a binary presence attribute (glasses,
        hat), a real feature can be occluded or hard to see in SOME
        frames (a head turn, glare, an awkward angle) without being
        genuinely absent - averaging would treat "hard to see this frame"
        identically to "absent this frame" and drag down an otherwise-
        confident detection. Deliberately NOT a proper probability
        distribution afterward (each label's score comes from whichever
        frame favored it most, so they don't have to sum to 1 the way one
        frame's own softmax output does) - that's intentional, each class
        gets to argue its own best case rather than being tied to
        whatever a single averaged embedding decided."""
        best_per_label = [0.0] * len(labels)
        for emb in image_embs:
            _, _, per_label = self._classify(emb, text_emb, labels)
            for i, label in enumerate(labels):
                best_per_label[i] = max(best_per_label[i], per_label[label])
        best_idx = int(np.argmax(best_per_label))
        return labels[best_idx], best_per_label[best_idx], dict(zip(labels, best_per_label))

    @staticmethod
    def _select_diverse(embs: list, scores: list, k: int, diversity_weight: float = 0.5) -> list:
        """Greedy Maximal Marginal Relevance (MMR): pick k candidates that
        are collectively the best trade-off between individually
        high-scoring AND mutually dissimilar in embedding space - not the
        k highest-scoring (which clusters on near-duplicate frames: a
        person holding one pose for a second generates several
        near-identical high-score frames in a row) and not the k most
        different (which would happily pick a blurry/occluded frame just
        because it looks unlike the others).

        Iterative: seed with the single highest-scoring candidate, then
        repeatedly add whichever remaining candidate maximizes
        `diversity_weight * score - (1 - diversity_weight) * max_similarity_to_any_already_selected`.
        Penalizing against the MAX similarity (not the average) to the
        selected set means a run of 5 near-identical high-score frames
        only ever contributes ONE of itself - each new pick has to be
        unlike EVERY frame already kept, not just unlike their average.

        embs are unit-normalized (see _embed_image), so a plain dot
        product IS the cosine similarity - no extra normalization needed.
        scores and similarities aren't rescaled to match ranges - both
        are already roughly unit-interval quantities in practice (this
        repo's score = face_conf * shoulder_score * sync_quality *
        size_quality, all in [0,1]; CLIP cosine similarities between
        real photos also live in a similar band) - a heuristic, not a
        rigorous joint calibration, but the intended one: this is a
        selection order, not a score anyone downstream ever sees."""
        n = len(embs)
        k = min(k, n)
        if k <= 0:
            return []
        remaining = list(range(n))
        first = max(remaining, key=lambda i: scores[i])
        selected = [first]
        remaining.remove(first)
        while len(selected) < k and remaining:
            best_i, best_mmr = None, None
            for i in remaining:
                max_sim = max(float(embs[i] @ embs[j].T) for j in selected)
                mmr = diversity_weight * scores[i] - (1 - diversity_weight) * max_sim
                if best_mmr is None or mmr > best_mmr:
                    best_i, best_mmr = i, mmr
            selected.append(best_i)
            remaining.remove(best_i)
        return selected

    @staticmethod
    def _pack_result(hair_color_result, clothing_color_result, glasses_result, hat_result):
        (hair_color, hair_color_conf, hair_color_boosted, hair_color_raw) = hair_color_result
        (clothing_color, clothing_color_conf, clothing_color_boosted, clothing_color_raw) = clothing_color_result
        (glasses_present, glasses_conf, glasses_scores) = glasses_result
        (hat_present, hat_conf, hat_scores) = hat_result
        attrs = {
            'hair_color': hair_color, 'hair_color_confidence': hair_color_conf,
            'clothing_color': clothing_color, 'clothing_color_confidence': clothing_color_conf,
            'glasses_present': bool(glasses_present), 'glasses_confidence': glasses_conf,
            'hat_present': bool(hat_present), 'hat_confidence': hat_conf,
        }
        # boosted is the number the decision was actually made on; raw is
        # what CLIP said before the neighbor-corroboration boost - keeping
        # both in the debug dump is what makes "did the boost fire, and by
        # how much" visible instead of theoretical.
        raw = (
            f'hair_color boosted: {hair_color_boosted}\n'
            f'hair_color raw:     {hair_color_raw}\n'
            f'clothing_color boosted: {clothing_color_boosted}\n'
            f'clothing_color raw:     {clothing_color_raw}\n'
            f'glasses: {glasses_scores}\n'
            f'hat: {hat_scores}'
        )
        return attrs, raw

    def _classify_hair_color(self, head_emb: torch.Tensor):
        """Bald gate first, color question only if it comes back False -
        see the _BALD_LABELS comment for why this has to be two decisions,
        not an 8th color competing in the same softmax. Returns the same
        4-tuple shape _classify_smoothed does (label, conf, boosted dict,
        raw dict) either way, so callers/_pack_result don't need to care
        which branch actually ran."""
        bald, bald_conf, bald_scores = self._classify(head_emb, self._bald_emb, self._bald_labels)
        if bald:
            return 'bald', bald_conf, bald_scores, bald_scores
        return self._classify_smoothed(head_emb, self._hair_color_emb, self._hair_color_labels, self._hair_color_kernel)

    def _classify_all(self, head_emb: torch.Tensor, torso_emb: torch.Tensor):
        """Single-embedding classification for all 4 attributes - used by
        describe() (genuinely one frame, no pooling question to ask) and
        nowhere in describe_ensemble, which needs different pooling per
        attribute type (see that method)."""
        hair_color_result = self._classify_hair_color(head_emb)
        clothing_color_result = self._classify_smoothed(
            torso_emb, self._clothing_color_emb, self._clothing_color_labels, self._clothing_color_kernel)
        glasses_result = self._classify(head_emb, self._glasses_emb, self._glasses_labels)
        hat_result = self._classify(head_emb, self._hat_emb, self._hat_labels)
        return self._pack_result(hair_color_result, clothing_color_result, glasses_result, hat_result)

    def describe(self, head_crop: np.ndarray, torso_crop: np.ndarray, return_timing: bool = False):
        """Single-frame classification. describe_ensemble is the actual
        entry point person_description.py uses now (with any number of
        frames, including 1) - this is kept for direct/isolated testing
        and as the one-frame building block describe_ensemble embeds."""
        timing = {}
        t0 = time.perf_counter()
        head_emb = self._embed_image(head_crop)
        torso_emb = self._embed_image(torso_crop)
        t1 = time.perf_counter()
        timing['embed_images'] = t1 - t0

        attrs, raw = self._classify_all(head_emb, torso_emb)
        t2 = time.perf_counter()
        timing['classify'] = t2 - t1
        timing['total'] = t2 - t0

        return (attrs, timing, raw) if return_timing else attrs

    def describe_ensemble(
            self, frame_pairs: list, ensemble_k: int = None,
            diversity_weight: float = 0.5, return_timing: bool = False):
        """frame_pairs: list of (head_crop, torso_crop, score) - a
        CANDIDATE POOL, typically larger than the number actually used
        for classification. score is whatever per-frame quality score
        the caller already computed (person_description.py's face/
        shoulder/sync/size product) - used to seed and weight the
        diversity selection below, not for pooling itself.

        ensemble_k (None = use every candidate, old behaviour): how many
        of the candidates to actually classify over, chosen via
        _select_diverse (MMR: score + head-embedding diversity) rather
        than plain top-k-by-score. Picked in response to a live-confirmed
        failure mode: top-k-by-score alone tends to keep several
        near-identical frames (same angle, same instant) when a person
        holds one pose, which does nothing for an ANGLE-dependent miss
        (glasses confirmed live to read differently depending on head
        angle) - diversity-aware selection is what actually gets
        different angles into the pool being classified. Selection runs
        on the HEAD embedding specifically (not torso) since that's
        where the angle-dependent attributes (glasses/hat) live.

        Different pooling per attribute type on the SELECTED subset, not
        one strategy for everything - confirmed live these two failure
        modes need opposite treatment:
        - hair_color/clothing_color: average embeddings across frames
          BEFORE classifying (_average_embeddings) - the same person's
          hair_color was confirmed live to flip between calls (wrong at
          ~0.56 confidence, correct at ~0.87) on the SAME checkpoint, no
          capacity difference to explain it, just per-frame noise. A
          wrong color read isn't announced by low confidence (bad
          lighting can be just as confident as good), so smoothing the
          signal before it hardens into a label is what actually helps.
        - glasses_present/hat_present: max-pool each frame's OWN
          classification (_classify_max_pool) - these have a real
          occlusion asymmetry averaging ignores (glasses can be genuinely
          present but hard to see in some frames without being absent),
          so each candidate should be scored by its single best-
          supporting frame instead of an averaged embedding that treats
          "occluded this frame" the same as "absent this frame"."""
        timing = {}
        t0 = time.perf_counter()
        head_embs = [self._embed_image(head) for head, _torso, _score in frame_pairs]
        torso_embs = [self._embed_image(torso) for _head, torso, _score in frame_pairs]
        scores = [score for _head, _torso, score in frame_pairs]
        t1 = time.perf_counter()
        timing['embed_images'] = t1 - t0

        if ensemble_k is not None and ensemble_k < len(frame_pairs):
            selected = self._select_diverse(head_embs, scores, ensemble_k, diversity_weight)
            head_embs = [head_embs[i] for i in selected]
            torso_embs = [torso_embs[i] for i in selected]
        else:
            selected = list(range(len(frame_pairs)))

        avg_head_emb = self._average_embeddings(head_embs)
        avg_torso_emb = self._average_embeddings(torso_embs)
        hair_color_result = self._classify_hair_color(avg_head_emb)
        clothing_color_result = self._classify_smoothed(
            avg_torso_emb, self._clothing_color_emb, self._clothing_color_labels, self._clothing_color_kernel)
        glasses_result = self._classify_max_pool(head_embs, self._glasses_emb, self._glasses_labels)
        hat_result = self._classify_max_pool(head_embs, self._hat_emb, self._hat_labels)
        attrs, raw = self._pack_result(hair_color_result, clothing_color_result, glasses_result, hat_result)
        t2 = time.perf_counter()
        timing['classify'] = t2 - t1
        timing['total'] = t2 - t0
        raw = (
            f'[MMR-selected {len(selected)}/{len(frame_pairs)} frame(s) (diversity_weight='
            f'{diversity_weight}): mean-pool color, max-pool binary]\n{raw}'
        )

        return (attrs, timing, raw) if return_timing else attrs
