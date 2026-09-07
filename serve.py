import asyncio
import logging
import os
import socket
import sys
import tempfile
import time
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from faster_whisper import BatchedInferencePipeline, WhisperModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [" + (os.getenv("INSTANCE_NAME") or socket.gethostname()) + "] %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_NAME    = os.getenv("MODEL_NAME", "large-v3")
DEVICE        = os.getenv("DEVICE", "cuda")
COMPUTE_TYPE  = os.getenv("COMPUTE_TYPE", "float16")
BEAM_SIZE     = int(os.getenv("BEAM_SIZE", "5"))
BATCH_SIZE    = int(os.getenv("BATCH_SIZE", "8"))
VAD_FILTER    = os.getenv("VAD_FILTER", "1") == "1"
DEFAULT_LANG  = os.getenv("DEFAULT_LANGUAGE", "en")
INCLUDE_PROB  = os.getenv("INCLUDE_WORD_PROBABILITY", "0") == "1"
HOTWORDS_FILE = os.getenv("HOTWORDS_FILE", "/app/hotwords.txt")
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "1"))
# Appears in /health, log lines, and the X-Whisper-Instance response header.
INSTANCE_NAME = os.getenv("INSTANCE_NAME") or socket.gethostname()

BATCHED = BATCH_SIZE > 1


def _load_hotwords() -> Optional[str]:
    if not os.path.exists(HOTWORDS_FILE):
        return None
    with open(HOTWORDS_FILE, encoding="utf-8") as fh:
        terms = [ln.strip() for ln in fh
                 if ln.strip() and not ln.lstrip().startswith("#")]
    if not terms:
        return None
    joined = ", ".join(terms)
    logger.info("Loaded %d hotword terms (%d chars)", len(terms), len(joined))
    return joined


HOTWORDS = _load_hotwords()

# Never serve from a CPU-only CTranslate2; the fallback is silent.
if DEVICE == "cuda":
    import ctranslate2
    n_gpu = ctranslate2.get_cuda_device_count()
    if n_gpu < 1:
        logger.error("ctranslate2 %s reports 0 CUDA devices - CPU-only build. Refusing to start.",
                     ctranslate2.__version__)
        sys.exit(1)
    logger.info("ctranslate2 %s, %d CUDA device(s), compute types: %s",
                ctranslate2.__version__, n_gpu,
                ctranslate2.get_supported_compute_types("cuda"))

model = WhisperModel(MODEL_NAME, device=DEVICE, device_index=0,
                     compute_type=COMPUTE_TYPE, num_workers=MAX_CONCURRENT)
batched = BatchedInferencePipeline(model=model) if BATCHED else None


def _transcribe(audio, *, language, beam_size, initial_prompt, hotwords,
                temperature, word_timestamps, vad_filter):
    kw = dict(
        language=language,
        beam_size=beam_size,
        best_of=beam_size,
        initial_prompt=initial_prompt,
        hotwords=hotwords,
        temperature=temperature,
        word_timestamps=word_timestamps,
        vad_filter=vad_filter,
        condition_on_previous_text=False,
    )
    # Batched decode needs VAD clips, so a no-VAD request goes sequential.
    if BATCHED and vad_filter:
        return batched.transcribe(audio, batch_size=BATCH_SIZE, **kw)
    return model.transcribe(audio, **kw)


# First call JIT-compiles kernels. word_timestamps warms the alignment path.
_t0 = time.perf_counter()
_segs, _ = _transcribe(
    np.zeros(16000, dtype=np.float32), language=DEFAULT_LANG, beam_size=BEAM_SIZE,
    initial_prompt=None, hotwords=None, temperature=0.0, word_timestamps=True,
    vad_filter=False,
)
list(_segs)
logger.info("Warm-up completed in %.2fs (model=%s, compute=%s, beam=%d, batch=%s)",
            time.perf_counter() - _t0, MODEL_NAME, COMPUTE_TYPE, BEAM_SIZE,
            BATCH_SIZE if BATCHED else "off")

app = FastAPI(title="faster-whisper OpenAI-compatible ASR")


@app.middleware("http")
async def tag_instance(request, call_next):
    response = await call_next(request)
    response.headers["X-Whisper-Instance"] = INSTANCE_NAME
    return response

# Scale out with replicas, not this. Executor keeps the event loop free.
_gpu_lock = asyncio.Semaphore(MAX_CONCURRENT)


def _normalize_granularities(raw: Optional[List[str]]) -> List[str]:
    """Accept both spellings, repeated fields, and comma-joined values."""
    if not raw:
        return ["segment"]
    out = []
    for item in raw:
        for part in str(item).split(","):
            part = part.strip().strip("[]'\"")
            if part in ("word", "segment") and part not in out:
                out.append(part)
    return out or ["segment"]


def _srt_time(t: float) -> str:
    h, rem = divmod(int(t), 3600)
    m, s = divmod(rem, 60)
    ms = int(round((t - int(t)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _vtt_time(t: float) -> str:
    return _srt_time(t).replace(",", ".")


@app.get("/health")
@app.get("/v1/health")
async def health():
    return {
        "status": "ok",
        "instance": INSTANCE_NAME,
        "model": MODEL_NAME,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "beam_size": BEAM_SIZE,
        "batch_size": BATCH_SIZE if BATCHED else None,
        "max_concurrent": MAX_CONCURRENT,
        "vad_filter": VAD_FILTER,
        "hotwords_loaded": HOTWORDS is not None,
    }


@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [
        {"id": "whisper-1", "object": "model", "owned_by": "local"},
        {"id": MODEL_NAME, "object": "model", "owned_by": "local"},
    ]}


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model_name: str = Form("whisper-1", alias="model"),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    timestamp_granularities: Optional[List[str]] = Form(None, alias="timestamp_granularities[]"),
    timestamp_granularities_alt: Optional[List[str]] = Form(None, alias="timestamp_granularities"),
    # Off-spec extensions
    hotwords: Optional[str] = Form(None),
    beam_size: Optional[int] = Form(None),
    vad_filter: Optional[bool] = Form(None),
):
    grans = _normalize_granularities(timestamp_granularities or timestamp_granularities_alt)
    want_words = "word" in grans

    if want_words and response_format != "verbose_json":
        raise HTTPException(
            status_code=400,
            detail="response_format must be 'verbose_json' to use timestamp_granularities",
        )

    effective_hotwords = hotwords if hotwords is not None else HOTWORDS
    effective_beam = beam_size or BEAM_SIZE
    effective_vad = VAD_FILTER if vad_filter is None else vad_filter

    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp.flush()
        tmp_path = tmp.name

    try:
        def _run():
            segs, inf = _transcribe(
                tmp_path,
                language=language or DEFAULT_LANG,
                beam_size=effective_beam,
                initial_prompt=prompt,
                hotwords=effective_hotwords,
                temperature=temperature,
                word_timestamps=want_words,
                vad_filter=effective_vad,
            )
            return list(segs), inf   # generator -> list; forces actual decode

        t_queued = time.perf_counter()
        async with _gpu_lock:
            t0 = time.perf_counter()
            segments, info = await asyncio.get_running_loop().run_in_executor(None, _run)
            elapsed = time.perf_counter() - t0
        queue_wait = t0 - t_queued
        rtf = (info.duration / elapsed) if elapsed > 0 else 0.0
        logger.info("Transcribed %.1fs in %.2fs (RTF %.1fx, queue_wait %.2fs, beam=%d, "
                    "batch=%s, words=%s, vad=%s, hotwords=%s)",
                    info.duration, elapsed, rtf, queue_wait, effective_beam,
                    BATCH_SIZE if (BATCHED and effective_vad) else "off",
                    want_words, effective_vad,
                    bool(effective_hotwords))

        text = "".join(s.text for s in segments).strip()

        if response_format == "text":
            return PlainTextResponse(text)

        if response_format == "srt":
            lines = []
            for i, s in enumerate(segments, 1):
                lines.append(f"{i}\n{_srt_time(s.start)} --> {_srt_time(s.end)}\n{s.text.strip()}\n")
            return PlainTextResponse("\n".join(lines))

        if response_format == "vtt":
            lines = ["WEBVTT\n"]
            for s in segments:
                lines.append(f"{_vtt_time(s.start)} --> {_vtt_time(s.end)}\n{s.text.strip()}\n")
            return PlainTextResponse("\n".join(lines))

        if response_format != "verbose_json":
            return JSONResponse({"text": text})

        out = {
            "task": "transcribe",
            "language": info.language,
            "duration": info.duration,
            "text": text,
        }

        def _words_of(s):
            return [{"word": w.word, "start": w.start, "end": w.end}
                    | ({"probability": w.probability} if INCLUDE_PROB else {})
                    for w in (s.words or [])]

        if "segment" in grans:
            out["segments"] = [{
                "id": i,
                "seek": getattr(s, "seek", 0),
                "start": s.start,
                "end": s.end,
                "text": s.text,
                "tokens": list(getattr(s, "tokens", []) or []),
                "temperature": getattr(s, "temperature", temperature),
                "avg_logprob": getattr(s, "avg_logprob", 0.0),
                "compression_ratio": getattr(s, "compression_ratio", 0.0),
                "no_speech_prob": getattr(s, "no_speech_prob", 0.0),
                # Off-spec: OpenAI puts words only at the top level.
                **({"words": _words_of(s)} if want_words else {}),
            } for i, s in enumerate(segments)]

        if want_words:
            out["words"] = [w for s in segments for w in _words_of(s)]

        return JSONResponse(out)

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
