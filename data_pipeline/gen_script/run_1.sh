export API_KEY=""
export MODEL_NAME=""
export BASEURL_POOL=""
export TIMEOUT_LIMIT=300
export CONCURRENCY_LIMIT=50

echo "[RUN] Starting all steps..."
python 1_1_main_entities.py --root_path <root_path> &

python 1_2_non_speech.py --root_path <root_path> &

python 1_3_transcribe.py --root_path <root_path> &

wait

echo "All steps have finished."
