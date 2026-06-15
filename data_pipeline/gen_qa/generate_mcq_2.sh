export API_KEY=""
export MODEL_NAME=""
export BASEURL_POOL=""
export TIMEOUT_LIMIT=300
export CONCURRENCY_LIMIT=50
export QA_NUM=2

echo "Starting all tasks in parallel..."

python generate_mcq_2.py --root_path <root_path> --task comparison &
python generate_mcq_2.py --root_path <root_path> --task sentiment_analysis &
python generate_mcq_2.py --root_path <root_path> --task summarization &
python generate_mcq_2.py --root_path <root_path> --task causal_reasoning &
python generate_mcq_2.py --root_path <root_path> --task future_prediction &
python generate_mcq_2.py --root_path <root_path> --task hypothetical_reasoning &

wait

python parse_mcq.py --root_path <root_path>

echo "All tasks have finished."