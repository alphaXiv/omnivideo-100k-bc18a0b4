export API_KEY=""
export MODEL_NAME=""
export BASEURL_POOL=""
export TIMEOUT_LIMIT=300
export CONCURRENCY_LIMIT=50
export QA_NUM=2

python generate_qa.py --root_path <root_path> --task fine_grained_perception
python parse_qa.py --root_path <root_path>
python generate_mcq_1.py --root_path <root_path> --task fine_grained_perception
python parse_mcq.py --root_path <root_path>

python generate_qa.py --root_path <root_path> --task scene_transformation_detection
python parse_qa.py --root_path <root_path>
python generate_mcq_1.py --root_path <root_path> --task scene_transformation_detection
python parse_mcq.py --root_path <root_path>

python generate_qa.py --root_path <root_path> --task context_understanding
python parse_qa.py --root_path <root_path>
python generate_mcq_1.py --root_path <root_path> --task context_understanding
python parse_mcq.py --root_path <root_path>

python generate_qa.py --root_path <root_path> --task event_sequence_ordering
python parse_qa.py --root_path <root_path>
python generate_mcq_1.py --root_path <root_path> --task event_sequence_ordering
python parse_mcq.py --root_path <root_path>
