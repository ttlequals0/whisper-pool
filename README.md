# whisper-pool

An OpenAI-compatible Whisper transcription server that runs several replicas
behind one endpoint, so a client can transcribe more than one thing at a time.

Built for NVIDIA DGX Spark (GB10, arm64, CUDA 13), where no working
`ctranslate2[cuda]` wheel exists and a plain `pip install faster-whisper`
silently gives you a CPU-only build. The image ships a CUDA build of
CTranslate2 with native sm_121 kernels and refuses to start if it ends up on
the CPU anyway.

## What you get

- `POST /v1/audio/transcriptions` matching the OpenAI Whisper API, including
  `verbose_json` and word-level timestamps
- Word timestamps in both shapes: a flat top-level `words[]` as OpenAI
  specifies, and nested inside each segment, which is what many clients read
- Contextual biasing through `hotwords`, applied across the whole file rather
  than just the first 30 seconds
- Every response tagged with `X-Whisper-Instance`, so you can tell which
  replica served a request through the load balancer
- A startup gate that exits rather than serving from a CPU-only build

## Quick start

Single replica:

    git clone https://github.com/ttlequals0/whisper-pool
    cd whisper-pool
    mkdir -p models
    docker compose -f examples/docker-compose.single.yml up -d

First start downloads `large-v3`, about 3 GB. Watch for the warm-up line:

    docker compose logs -f

    INFO [whisper] ctranslate2 4.6.0, 1 CUDA device(s), compute types: {...}
    INFO [whisper] Loaded 11 hotword terms (99 chars)
    INFO [whisper] Warm-up completed in 5.33s (model=large-v3, compute=float16, beam=5, batch=16)

A warm-up over 10 seconds usually means it fell back to the CPU.

Then transcribe:

    curl -s http://localhost:9000/v1/audio/transcriptions \
      -F "file=@audio.wav" \
      -F "model=whisper-1" \
      -F "response_format=verbose_json" \
      -F "timestamp_granularities[]=word" | jq '.words[:5]'

## Running a pool

    mkdir -p models
    # Seed the cache once so three replicas do not race on the same download.
    docker run --rm --gpus all -v "$PWD/models":/root/.cache/huggingface \
      ttlequals0/faster-whisper-dgx:1.0.2 \
      python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3', device='cuda', compute_type='float16')"

    docker compose up -d

That starts `whisper-1`, `whisper-2` and `whisper-3` on ports 9002 to 9004,
each serving one decode at a time. Put nginx in front of them
(`nginx/whisper-pool.conf`) and clients see a single endpoint.

Check the pool:

    for p in 9002 9003 9004; do curl -s localhost:$p/health | jq -r .instance; done

## How many replicas

Two separate limits, and it is worth knowing which one you are hitting.

**Per replica.** `MAX_CONCURRENT` is how many decodes one container runs at
once. Leave it at 1. Raising it shares one process and one CUDA context
across every slot, so a crash or a CUDA error takes them all down together.
Every deploy becomes a full outage too.

**Across replicas.** More replicas raise total throughput until the GPU runs
out of memory bandwidth. Whisper decoding is bandwidth bound rather than
compute bound, so the ceiling arrives sooner than the GPU's utilization
percentage suggests. On a DGX Spark:

| Concurrent decodes | Aggregate | Per request |
|---|---|---|
| 1 | 45.6x realtime | 45.6x |
| 2 | 51.7x | 25.8x |
| 3 | 55.1x | 18.4x |
| 4 | 55.3x | 13.8x |

Throughput flattens at three. A fourth replica adds latency to every request
and buys 0.2x. Find your own knee before settling on a number: it depends on
the GPU's memory bandwidth, not on how many cores it has.

Three separate containers measured the same as one container with three
workers (43.9x against 49.0x, and 49.4x with CUDA MPS enabled), so the
isolation is close to free.

## Configuration

All settings are environment variables.

| Variable | Default | Notes |
|---|---|---|
| `MODEL_NAME` | `large-v3` | Any faster-whisper model name |
| `DEVICE` | `cuda` | `cpu` skips the CUDA startup gate |
| `COMPUTE_TYPE` | `float16` | `int8_float16` is faster and less accurate |
| `BEAM_SIZE` | `5` | 1 is greedy: faster, more repetition loops |
| `BATCH_SIZE` | `8` | Batched decode inside one request. 16 is a good default on a modern GPU |
| `MAX_CONCURRENT` | `1` | Decodes at once in this container. See above |
| `VAD_FILTER` | `1` | Skips silence. Also suppresses very quiet speech |
| `DEFAULT_LANGUAGE` | `en` | Pin it. Auto-detect misfires on music intros |
| `INCLUDE_WORD_PROBABILITY` | `0` | Adds `probability` per word. Off-spec, so strict clients may reject it |
| `HOTWORDS_FILE` | `/app/hotwords.txt` | One term per line, `#` comments allowed |
| `INSTANCE_NAME` | container hostname | Appears in `/health`, logs, and the response header |

### Hotwords

`hotwords.txt` biases decoding toward terms the model gets wrong, across the
whole file. Write each term as it should appear in the output. Include expansions
next to acronyms when both get spoken:

    Redfish
    BMC
    baseboard management controller

Published results put rare-word error at 23.7% falling to 18.0% with
contextual biasing, and out-of-vocabulary error at 60.0% falling to 37.1%.
Measure precision as well as recall: biasing can also insert a hot term where
it was never said.

Callers can override the file per request with a `hotwords` form field.

## API

### `POST /v1/audio/transcriptions`

Standard OpenAI fields: `file`, `model`, `language`, `prompt`,
`response_format` (`json`, `text`, `srt`, `vtt`, `verbose_json`),
`temperature`, `timestamp_granularities[]`.

Extensions beyond the OpenAI schema, all optional:

| Field | Effect |
|---|---|
| `hotwords` | Comma-separated terms, replacing `hotwords.txt` for this request |
| `beam_size` | Override the configured beam size |
| `vad_filter` | `false` disables VAD for this request |

`vad_filter=false` matters for recovering quiet audio that VAD drops, such as
a soft outro at the end of a file. Batched decoding derives its clips from VAD
and cannot run without it, so a no-VAD request falls back to sequential
decoding automatically. Nothing to configure; short clips cost nothing this
way.

Word timestamps require `response_format=verbose_json`. They are returned
twice, at the top level and inside each segment:

    {
      "task": "transcribe",
      "language": "en",
      "duration": 11.0,
      "text": "And so my fellow Americans...",
      "segments": [
        {"id": 0, "start": 0.0, "end": 11.0, "text": "...",
         "words": [{"word": " And", "start": 0.0, "end": 0.44}]}
      ],
      "words": [{"word": " And", "start": 0.0, "end": 0.44}]
    }

Words keep their leading space, matching both OpenAI and faster-whisper. Do
not strip it if you rebuild text by concatenation.

### `GET /health` and `GET /v1/health`

    {"status":"ok","instance":"whisper-1","model":"large-v3","device":"cuda",
     "compute_type":"float16","beam_size":5,"batch_size":16,
     "max_concurrent":1,"vad_filter":true,"hotwords_loaded":true}

Both paths work. `/v1/health` exists because monitoring tools tend to append
the version prefix to everything.

### `GET /v1/models`

Reports `whisper-1` and the loaded model name, so SDKs that list models before
transcribing do not fail.

## Client notes

Any OpenAI SDK works by pointing `base_url` at the endpoint:

    from openai import OpenAI

    client = OpenAI(base_url="http://localhost:9000/v1", api_key="not-needed")
    with open("audio.wav", "rb") as f:
        r = client.audio.transcriptions.create(
            file=f, model="whisper-1",
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"],
        )

The SDK cannot send the off-spec fields. Use `extra_body` or plain HTTP for
those.

There is no authentication. The server is meant to sit on a private network or
behind a proxy that handles auth. Do not expose it directly.

**Concurrency belongs to the client.** The pool accepts as many simultaneous
requests as it has replicas; beyond that, requests queue inside a replica and
wait. Track `queue_wait` in the logs. A consistent 0.00 means you are within
capacity; a rising value means the client is asking for more than the pool has.

## Building

The published image is `linux/arm64`, built for DGX Spark. To build it
yourself:

    docker build -t whisper-pool .

The image installs a CUDA build of the CTranslate2 C++ library from a release
tarball, then recompiles the Python bindings against it. That second step is not
optional. The PyPI wheel is built with `_GLIBCXX_USE_CXX11_ABI=0` and the CUDA
library uses the new ABI, so dropping in the shared object on its own fails with
an undefined `std::string` symbol.

For other architectures, replace `CT2_URL` with a CUDA build for your platform
and set `CMAKE_CUDA_ARCHITECTURES` to match your GPU.

## Troubleshooting

**Refuses to start, logs `0 CUDA devices - CPU-only build`.** The container has
no GPU, or CTranslate2 is a CPU build. Check `docker run --rm --gpus all <image>
nvidia-smi` and that the compose file has the `deploy.resources` block. Note
`runtime: nvidia` is not recognized on some hosts, including DGX OS.

**All traffic lands on one replica.** The nginx `upstream` is missing its
`zone` directive, so load balancing state is per worker and each worker picks
the first server independently.

**Some requests fail through the proxy but every replica answers directly.**
Look for nginx workers stuck in `shutting down`. They keep serving the old
configuration until their connections drain, which for long uploads can be
hours. Restart nginx rather than reloading it.

**Requests are slow but the GPU looks idle.** Whisper decoding is bandwidth
bound. High utilization at low power draw means the GPU is waiting on memory,
and adding replicas will not help.

**A short clip returns 200 with no segments.** VAD suppressed it. Retry that
request with `vad_filter=false`.

## License

MIT
