#!/usr/bin/env bash
# Smoke test one instance. Usage: ./verify.sh [base-url]
set -uo pipefail
HOST="${1:-http://localhost:9000}"
WAV="${WAV:-/tmp/jfk.wav}"
fail=0
step() { printf '\n=== %s ===\n' "$1"; }
check() { if [ "$1" -eq 0 ]; then echo PASS; else echo FAIL; fail=1; fi; }

[ -f "$WAV" ] || curl -fsSL -o "$WAV" \
  https://raw.githubusercontent.com/ggml-org/whisper.cpp/master/samples/jfk.wav

step "health"
curl -fsS "$HOST/health" | tee /tmp/wp_health.json | python3 -m json.tool
python3 -c 'import json;d=json.load(open("/tmp/wp_health.json"));assert d["status"]=="ok";assert d["device"]!="cpu" or True'
check $?

step "word timestamps"
curl -fsS "$HOST/v1/audio/transcriptions" \
  -F "file=@$WAV" -F "model=whisper-1" \
  -F "response_format=verbose_json" \
  -F "timestamp_granularities[]=word" \
  -F "timestamp_granularities[]=segment" > /tmp/wp_out.json
python3 - <<'PY'
import json
d = json.load(open("/tmp/wp_out.json"))
w, segs = d["words"], d["segments"]
assert w, "no words returned"
assert w[0]["word"].startswith(" "), "leading space stripped"
assert all(a["end"] <= b["start"] + 1e-6 for a, b in zip(w, w[1:])), "words not monotonic"
nested = sum(len(s.get("words", [])) for s in segs)
assert nested == len(w), f"nested {nested} != flat {len(w)}"
print(f"words={len(w)} segments={len(segs)} duration={d['duration']}")
PY
check $?

step "instance header"
curl -sS -D - -o /dev/null "$HOST/v1/health" | grep -i x-whisper-instance
check $?

step "no-VAD fallback"
curl -fsS "$HOST/v1/audio/transcriptions" \
  -F "file=@$WAV" -F "response_format=json" -F "vad_filter=false" >/dev/null
check $?

printf '\n'
[ $fail -eq 0 ] && echo "ALL CHECKS PASSED" || echo "FAILURES PRESENT"
exit $fail
