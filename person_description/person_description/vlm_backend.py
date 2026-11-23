"""Qwen2.5-VL wrapper: one image crop -> structured appearance attributes.

Kept as its own module (not inlined in person_description.py) so the ROS
node stays importable without torch/transformers on the path (tests,
`--help`, load_model:=false wiring checks), and so swapping the backend
later (different VLM, or a classical dominant-color cross-check for
clothing_color specifically) is a one-file change.
"""

import json
import re
import time

import numpy as np
from PIL import Image as PILImage

_MODEL_ID = 'Qwen/Qwen2.5-VL-3B-Instruct'

_PROMPT = (
    'Look at this cropped photo of one person. Respond with ONLY a single '
    'JSON object, no other text, no markdown code fences, with exactly these keys:\n'
    '{"hair_color": one of ["black","brown","blonde","gray","red","unknown"], '
    '"hair_color_confidence": float 0-1, '
    '"clothing_color": the single dominant color word of their main visible '
    'garment (e.g. "blue", "black", "red"), '
    '"clothing_color_confidence": float 0-1, '
    '"glasses_present": true or false, '
    '"glasses_confidence": float 0-1, '
    '"hat_present": true or false, '
    '"hat_confidence": float 0-1}\n'
    'Confidence reflects how clearly visible/certain that specific attribute '
    'is in THIS image - use a LOW confidence (below 0.5) for anything '
    'occluded, blurry, or ambiguous rather than guessing.\n'
    'Example of the exact syntax required (these are placeholder values '
    'only, not a hint about the actual answer - use the real image):\n'
    '{"hair_color": "black", "hair_color_confidence": 0.82, '
    '"clothing_color": "blue", "clothing_color_confidence": 0.9, '
    '"glasses_present": false, "glasses_confidence": 0.6, '
    '"hat_present": false, "hat_confidence": 0.7}'
)

_DEFAULT_ATTRS = {
    'hair_color': 'unknown', 'hair_color_confidence': 0.0,
    'clothing_color': 'unknown', 'clothing_color_confidence': 0.0,
    'glasses_present': False, 'glasses_confidence': 0.0,
    'hat_present': False, 'hat_confidence': 0.0,
}


class DescribeVLM:
    def __init__(self, device: str = 'cuda'):
        t0 = time.perf_counter()
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
        t1 = time.perf_counter()

        self._torch = torch
        self._processor = AutoProcessor.from_pretrained(_MODEL_ID)
        t2 = time.perf_counter()
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            _MODEL_ID, quantization_config=quant, device_map=device,
        )
        self._model.eval()
        t3 = time.perf_counter()

        self.load_timing = {
            'import': t1 - t0, 'processor': t2 - t1, 'model_weights': t3 - t2, 'total': t3 - t0,
        }
        print(f'DescribeVLM load timing: ' +
              ' '.join(f'{k}={v * 1000:.0f}ms' for k, v in self.load_timing.items()))

    def describe(self, bgr_crop: np.ndarray, return_timing: bool = False):
        timing = {}
        t0 = time.perf_counter()

        rgb = bgr_crop[:, :, ::-1]
        pil_img = PILImage.fromarray(rgb)
        messages = [{
            'role': 'user',
            'content': [
                {'type': 'image', 'image': pil_img},
                {'type': 'text', 'text': _PROMPT},
            ],
        }]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(text=[text], images=[pil_img], return_tensors='pt').to(self._model.device)
        t1 = time.perf_counter()
        timing['preprocess'] = t1 - t0

        with self._torch.no_grad():
            generated = self._model.generate(**inputs, max_new_tokens=200, do_sample=False)
        t2 = time.perf_counter()
        timing['generate'] = t2 - t1

        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
        raw = self._processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        attrs = _parse_attrs(raw)
        t3 = time.perf_counter()
        timing['decode_parse'] = t3 - t2
        timing['total'] = t3 - t0

        return (attrs, timing, raw) if return_timing else attrs


def _parse_attrs(raw: str) -> dict:
    # The model sometimes emits a malformed JSON-ish fragment before the
    # real, well-formed object (seen live: a garbled leading '{"A: and
    # hair content medium...' followed by the actual correct JSON). A
    # single greedy first-'{'-to-last-'}' match swallows both and fails
    # to parse, so instead: our schema is flat (no nested braces), so
    # scan every non-nested {...} span and take the LAST one that both
    # parses and contains at least one expected key - the model's real
    # answer is consistently the final complete object it emits.
    candidates = re.findall(r'\{[^{}]*\}', raw)
    for candidate in reversed(candidates):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict) or not (set(parsed) & set(_DEFAULT_ATTRS)):
            continue
        attrs = dict(_DEFAULT_ATTRS)
        attrs.update({k: parsed[k] for k in _DEFAULT_ATTRS if k in parsed})
        return attrs
    return _loose_extract(raw)


def _loose_extract(raw: str) -> dict:
    """Last-resort fallback for when no candidate span parses as valid
    JSON at all. Seen live on this exact 4-bit-quantized 3B model: the
    object is consistently near-miss rather than pure garbage - right
    keys, right values, but a stray non-JSON line dropped in (a leaked
    training artifact, e.g. "A: and B are mutually exclusive") or one
    value missing its opening quote. Requiring the WHOLE object to parse
    throws away an otherwise-good answer over one broken character, so
    instead extract each of the 8 fields independently - tolerant of
    stray surrounding text, missing keys (left at the unknown/0.0
    default), and a missing/extra quote on the value itself."""
    attrs = dict(_DEFAULT_ATTRS)
    for key in ('hair_color', 'clothing_color'):
        m = re.search(rf'"{key}"\s*:\s*"?([a-zA-Z]+)"?', raw)
        if m:
            attrs[key] = m.group(1).lower()
    for key in ('hair_color_confidence', 'clothing_color_confidence',
                'glasses_confidence', 'hat_confidence'):
        m = re.search(rf'"{key}"\s*:\s*([0-9]*\.?[0-9]+)', raw)
        if m:
            attrs[key] = float(m.group(1))
    for key in ('glasses_present', 'hat_present'):
        m = re.search(rf'"{key}"\s*:\s*"?(true|false)"?', raw, re.IGNORECASE)
        if m:
            attrs[key] = m.group(1).lower() == 'true'
    return attrs
