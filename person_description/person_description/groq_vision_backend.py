"""Groq-hosted vision inference for /perception/people/describe - alternative
to the local Qwen2.5-VL-3B-4bit backend (vlm_backend.py). Selected via the
`backend` ROS 2 parameter (backend:=groq). Same public interface as
DescribeVLM (describe(bgr_crop, return_timing=True) -> (attrs, timing, raw))
so person_description.py needs no changes beyond which module it constructs.

Picked over the local backend because the local 4-bit-quantized 3B model was
confirmed live (this repo's own testing) to produce both malformed JSON
("A: and B are mutually exclusive" leaking into the object, unclosed
braces) and confidently-wrong attribute values (hair_color=brown at 0.9
confidence for an actually-black-haired subject). A Groq-hosted 27B model
plus its json_mode (response_format={"type": "json_object"}, forces
syntactically valid JSON at the API level - confirmed supported via
GET https://api.groq.com/openai/v1/models) fixes the formatting failure
outright rather than patching around it with regex fallbacks, and removes
the local GPU/VRAM load entirely.

Model id verified live against the account's own Groq endpoint, not just
docs (docs/availability drift over time - same caveat llm_api_bridge's own
GroqProvider already notes): qwen/qwen3.6-27b, input_modalities
["text","image"], supported_features includes "json_mode".

GROQ_API_KEY is read HERE, not passed in from person_description.py - the
secret lives in this repo, next to the code that actually calls the LLM,
not injected by the dashboard or any shell rc file (same convention as
llm_api_bridge's own GroqProvider usage). .env is gitignored; .env.example
(tracked) documents the expected key.
"""

import base64
import json
import os
import time
from io import BytesIO

import numpy as np
from dotenv import load_dotenv
from PIL import Image as PILImage

from person_description.vlm_backend import _DEFAULT_ATTRS, _PROMPT

def _load_env_file() -> None:
    explicit = os.environ.get('PERSON_DESCRIPTION_ENV_FILE')
    if explicit and os.path.isfile(explicit):
        load_dotenv(explicit)
        return
    cur = os.path.dirname(os.path.realpath(__file__))
    for _ in range(5):
        candidate = os.path.join(cur, '.env')
        if os.path.isfile(candidate):
            load_dotenv(candidate)
            return
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    load_dotenv()


_load_env_file()


class GroqVisionDescribeVLM:
    def __init__(self, model: str = 'qwen/qwen3.6-27b'):
        # Imported lazily, same reasoning as vlm_backend.DescribeVLM's own
        # lazy transformers import - the ROS node stays importable without
        # the `openai` package unless this backend is actually selected.
        from openai import OpenAI

        # Missing key is not fatal here - matches llm_api_bridge's own
        # GroqProvider (api_key or "unset"): the caller (person_description.py)
        # warns via its ROS logger, construction still succeeds, and the
        # actual API call fails cleanly later (caught by _on_describe's
        # try/except) rather than crashing node startup on eager_load:=true.
        api_key = os.environ.get('GROQ_API_KEY')
        self._model = model
        self._client = OpenAI(
            api_key=api_key or 'unset',
            base_url='https://api.groq.com/openai/v1',
            timeout=20.0,
            max_retries=0,
        )
        # No local model weights to load - kept only so callers that log
        # load_timing (person_description.py's _ensure_vlm_loaded) don't
        # need a branch for which backend is active.
        self.load_timing = {'total': 0.0}

    def describe(self, bgr_crop: np.ndarray, return_timing: bool = False):
        timing = {}
        t0 = time.perf_counter()

        rgb = bgr_crop[:, :, ::-1]
        buf = BytesIO()
        PILImage.fromarray(rgb).save(buf, format='JPEG', quality=90)
        b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        t1 = time.perf_counter()
        timing['encode'] = t1 - t0

        completion = self._client.chat.completions.create(
            model=self._model,
            messages=[{
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': _PROMPT},
                    {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b64}'}},
                ],
            }],
            response_format={'type': 'json_object'},
            max_tokens=400,
            # qwen/qwen3.6-27b is a reasoning model - by default it spends
            # a large hidden <think>...</think> pass before ever reaching
            # the JSON answer. Confirmed live against the REAL 8-field
            # _PROMPT: reasoning_tokens=1197 with reasoning left on, which
            # both needs max_tokens>=~1300 just to not get cut off
            # mid-reasoning (finish_reason="length", empty content, 400
            # json_validate_failed) AND single-handedly blows this
            # account's 1000 output-tokens-per-minute cap in one request
            # (429 rate_limit_exceeded) - the two requirements are
            # mutually exclusive at reasoning-on. reasoning_effort='none'
            # (extra_body - not a typed SDK kwarg, and this model only
            # accepts 'none'/'default', not 'low'/'medium'/'high') drops
            # reasoning_tokens to 0 entirely: confirmed live, 93
            # completion tokens total, 0.3s round trip, well inside the
            # rate limit. Tested only against a synthetic no-person image
            # so far - worth confirming this doesn't cost real accuracy
            # against an actual face/body crop before trusting it fully.
            extra_body={'reasoning_format': 'hidden', 'reasoning_effort': 'none'},
        )
        t2 = time.perf_counter()
        timing['api_call'] = t2 - t1

        raw = completion.choices[0].message.content or ''
        attrs = _parse_json_response(raw)
        t3 = time.perf_counter()
        timing['parse'] = t3 - t2
        timing['total'] = t3 - t0

        return (attrs, timing, raw) if return_timing else attrs


def _parse_json_response(raw: str) -> dict:
    # json_mode guarantees syntactically valid JSON, but not that every
    # expected key is present - still fill in _DEFAULT_ATTRS for anything
    # missing, same contract as vlm_backend._parse_attrs.
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return dict(_DEFAULT_ATTRS)
    if not isinstance(parsed, dict):
        return dict(_DEFAULT_ATTRS)
    attrs = dict(_DEFAULT_ATTRS)
    attrs.update({k: parsed[k] for k in _DEFAULT_ATTRS if k in parsed})
    return attrs
