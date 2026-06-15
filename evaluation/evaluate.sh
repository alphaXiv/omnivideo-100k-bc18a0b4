python evaluation.py \
    --dataset "omnivideo_test" \
    --dataset_dir "OmniVideo-Test" \
    --model_type "qwen25_omni" \
    --model_path "Qwen/Qwen2.5-Omni-7B"

export HF_HOME="/data1/caixinyue/models"
export HF_ENDPOINT="https://hf-mirror.com"
export CUDA_VISIBLE_DEVICES="7"
export PYTHONPATH=$PYTHONPATH:/home/caixinyue/omni_model/video-SALMONN-2/video_SALMONN2_plus
/home/caixinyue/anaconda3/envs/video_salmonn/bin/python evaluation.py \
    --dataset "omnivideo_test" \
    --dataset_dir "/data1/caixinyue/OmniVideo-Test" \
    --model_type "video_salmonn2_plus" \
    --model_path "tsinghua-ee/video-SALMONN2_plus_7B_full"

python evaluation_gemini.py \
    --dataset "omnivideo_test" \
    --dataset_dir "OmniVideo-Test" \
    --model_name "gemini-3.1-pro-preview" \
    --api_key "" \
    --base_url ""
