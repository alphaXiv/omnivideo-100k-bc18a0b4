#!/usr/bin/env bash
# Minimal, end-to-end reproduction of the OmniVideo-100K data engine
# (arXiv 2606.14702) on CPU.
#
# Runs the released pipeline UNMODIFIED:
#   data_pipeline/gen_script/0..5  -> Entity-Anchored Video Scripting -> script.jsonl
#   data_pipeline/gen_qa/generate_qa.py -> Clue-Guided QA Generation
# on two real OmniVideo-Test clips, then a direct single-pass QA baseline, and
# scores the paper's two core mechanisms.
#
# The engine calls Gemini via google-genai. We have an OpenRouter key (not a
# Gemini key), and OpenRouter serves Gemini with video+audio input, so
# repro/vendor/google/genai is a shim that forwards generate_content() to
# OpenRouter. The pipeline scripts and prompts are untouched.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
export REPO_ROOT
export ROOT_PATH="$REPO_ROOT/omni_data"
mkdir -p "$ROOT_PATH"

# --- model / API config -------------------------------------------------------
: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY}"
export API_KEY="$OPENROUTER_API_KEY"
export MODEL_NAME="${MODEL_NAME:-google/gemini-2.5-flash}"
export BASEURL_POOL="${BASEURL_POOL:-https://openrouter.ai/api/v1}"  # shim ignores value; must be splittable
export CONCURRENCY_LIMIT="${CONCURRENCY_LIMIT:-4}"
export TIMEOUT_LIMIT="${TIMEOUT_LIMIT:-600}"
export OR_HTTP_TIMEOUT="${OR_HTTP_TIMEOUT:-600}"
export QA_NUM="${QA_NUM:-2}"
export PYTHONPATH="$REPO_ROOT/repro/vendor:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

CROSS_TASKS=(causal_reasoning comparison summarization sentiment_analysis event_sequence_ordering future_prediction hypothetical_reasoning)

# --- system deps: ffmpeg + ffprobe -------------------------------------------
if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
  echo "[setup] installing ffmpeg via apt"
  (apt-get update -y && apt-get install -y --no-install-recommends ffmpeg) >/tmp/apt.log 2>&1 || true
fi
if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
  echo "[setup] apt failed; fetching static ffmpeg build"
  curl -fsSL https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz -o /tmp/ff.tar.xz
  mkdir -p /tmp/ff && tar -xf /tmp/ff.tar.xz -C /tmp/ff --strip-components=1
  install -m755 /tmp/ff/ffmpeg /tmp/ff/ffprobe /usr/local/bin/
fi
command -v ffmpeg ffprobe || { echo "[fatal] ffmpeg/ffprobe unavailable"; exit 1; }

# --- python 3.12 (repo uses 3.12-only f-string syntax) ------------------------
if ! command -v uv >/dev/null 2>&1; then
  echo "[setup] installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
uv venv --python 3.12 /tmp/ov_venv
# shellcheck disable=SC1091
source /tmp/ov_venv/bin/activate
uv pip install -q tqdm aiofiles
python --version

# --- step 0: fetch real OmniVideo clips, build the video list -----------------
echo "=== prepare: download videos ==="
python repro/prepare.py

GS=data_pipeline/gen_script
echo "=== 0: separate audio/video + downsample ==="
python "$GS/0_seprate_av.py" --root_path "$ROOT_PATH" --num_processes 2 --target_fps 1 --target_max_dim 480
echo "=== 1_1: main entities (video) ==="
python "$GS/1_1_main_entities.py" --root_path "$ROOT_PATH"
echo "=== 1_2: non-speech sounds (audio) ==="
python "$GS/1_2_non_speech.py"   --root_path "$ROOT_PATH"
echo "=== 1_3: transcription (audio) ==="
python "$GS/1_3_transcribe.py"   --root_path "$ROOT_PATH"
echo "=== 2_1: speaker labels (video+audio) ==="
python "$GS/2_1_label_speaker.py" --root_path "$ROOT_PATH"
echo "=== 2_2: video summary ==="
python "$GS/2_2_video_summary.py" --root_path "$ROOT_PATH"
echo "=== 3: integrate segments ==="
python "$GS/3_inte_seg.py" --root_path "$ROOT_PATH" --max_seg_length 15
echo "=== 4: per-segment visual descriptions ==="
python "$GS/4_seg_visual.py" --root_path "$ROOT_PATH" --num_chunks 2
echo "=== 5: finalize script.jsonl ==="
python "$GS/5_check_script.py" --root_path "$ROOT_PATH"

if [ ! -s "$ROOT_PATH/script.jsonl" ]; then
  echo "[fatal] script.jsonl was not produced; Entity-Anchored Scripting failed"
  exit 1
fi
echo "[ok] script.jsonl has $(wc -l < "$ROOT_PATH/script.jsonl") video script(s)"

# --- Clue-Guided QA generation + direct baseline ------------------------------
GQ=data_pipeline/gen_qa
for t in "${CROSS_TASKS[@]}"; do
  echo "=== clue-guided QA: $t ==="
  python "$GQ/generate_qa.py" --root_path "$ROOT_PATH" --task "$t"
  echo "=== direct (single-pass) QA: $t ==="
  python repro/direct_qa.py --root_path "$ROOT_PATH" --task "$t"
done

# --- score + artifacts --------------------------------------------------------
echo "=== metrics ==="
python repro/metrics.py
echo "=== DONE ==="
