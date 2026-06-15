export API_KEY=""
export MODEL_NAME=""
export BASEURL_POOL=""
export TIMEOUT_LIMIT=300
export CONCURRENCY_LIMIT=50

echo "[RUN] Starting all steps..."

python 2_1_label_speaker.py --root_path <root_path> &

python 2_2_video_summary.py --root_path <root_path> &

wait

python 3_inte_seg.py --root_path <root_path>

wait

echo "All folders have finished."
