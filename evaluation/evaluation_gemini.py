import os
import time
import json
import asyncio
import argparse
import aiofiles
from tqdm import tqdm
from google import genai
from google.genai import types
from datasets import Video_MME, Daily_Omni, OmniVideoBench, JointAVBench, FutureOmni, OmniVideo_Test, Video_MME_v2


TIMEOUT_LIMIT = 600
CONCURRENCY_LIMIT = 50


async def call_api(client, model_name, video_bytes, prompt, timeout):
    def sync_call():
        return client.models.generate_content(
            model=model_name,
            contents=types.Content(
                parts=[
                    types.Part(inline_data=types.Blob(data=video_bytes, mime_type='video/mp4')),
                    types.Part(text=prompt)
                ]
            ),
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    include_thoughts=True
                )
            )
        )
    
    try:
        response = await asyncio.wait_for(asyncio.to_thread(sync_call), timeout=timeout)
        return response
    except asyncio.TimeoutError:
        return None


async def process_single_video_concurrent(item, client, model_name, semaphore, file_lock, file_handle):
    async with semaphore:
        try:
            with open(item["video_path"], "rb") as f:
                video_bytes = await asyncio.to_thread(f.read)
        except:
            print(f"[ERROR] Cannot read video: {item["video_path"]}")
            return None

        try:
            run_start = time.time()

            response = await call_api(client, model_name, video_bytes, item["prompt"], timeout=TIMEOUT_LIMIT)
            if not response:
                print("[TIMEOUT] API call exceeded timeout:", TIMEOUT_LIMIT)
                return

            item["model_answer"] = response.text
            item["model_answer"] = item["model_answer"].replace("*", "")

            run_end = time.time()
            tokens_data = response.usage_metadata
            item.setdefault("run_data", {})
            item["run_data"] = {
                "input": tokens_data.prompt_token_count,
                "output": tokens_data.candidates_token_count,
                "thinking": tokens_data.thoughts_token_count,
                "cost_time": run_end - run_start
            }
                
            thoughts =[]
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if getattr(part, "thought", False) and part.text:
                        thoughts.append(part.text)
            if thoughts:
                item["model_thoughts"] = "\n".join(thoughts)
            else:
                item["model_thoughts"] = None

            async with file_lock:
                await file_handle.write(json.dumps(item) + "\n")
                await file_handle.flush()
            print("[SUCESS]", item["question_id"])
            return

        except Exception as e:
            print(f"[FAILED]: Error on {item["question_id"]} {e}")

        return


async def main(args, dataset):
    client = genai.Client(api_key=args.api_key,
                          http_options={"base_url": args.base_url})

    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    file_lock = asyncio.Lock()
    async with aiofiles.open(args.results_file, "a", encoding="utf-8") as f:
        tasks = []
        for item in dataset.unprocessed_data:
            tasks.append(asyncio.create_task(process_single_video_concurrent(item, client, args.model_name, semaphore, file_lock, f)))

        if tasks:
            for f_task in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
                await f_task

    dataset.calculate_accuracy()
    os._exit(0)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--dataset", type=str, choices=["video_mme", "daily_omni", "omnivideobench", "jointavbench", "futureomni", "omnivideo_test", "video_mme_v2"], required=True, help="Dataset to use for evaluation.")
    args.add_argument("--dataset_dir", type=str, required=True, help="Path to the dataset directory.")
    args.add_argument("--model_name", type=str, choices=["gemini-3.1-pro-preview"], required=True, help="Model type to use for evaluation.")
    args.add_argument("--api_key", type=str, required=True, help="API KEY for gemini")
    args.add_argument("--base_url", type=str, required=True, help="BASE_URL for gemini")
    args.add_argument("--results_file", type=str, default=None, help="Path to save the evaluation results.")
    args = args.parse_args()

    if not args.results_file: 
        args.results_file = os.path.join(f"results/{args.dataset}/{args.model_name.split('/')[-1]}.jsonl")
    os.makedirs(os.path.dirname(args.results_file), exist_ok=True)
    
    if args.dataset == "video_mme":
        dataset = Video_MME(dataset_dir=args.dataset_dir, results_file=args.results_file)
    elif args.dataset == "daily_omni":
        dataset = Daily_Omni(dataset_dir=args.dataset_dir, results_file=args.results_file)
    elif args.dataset == "omnivideobench":
        dataset = OmniVideoBench(dataset_dir=args.dataset_dir, results_file=args.results_file)
    elif args.dataset == "jointavbench":
        dataset = JointAVBench(dataset_dir=args.dataset_dir, results_file=args.results_file)
    elif args.dataset == "futureomni":
        dataset = FutureOmni(dataset_dir=args.dataset_dir, results_file=args.results_file)
    elif args.dataset == "omnivideo_test":
        dataset = OmniVideo_Test(dataset_dir=args.dataset_dir, results_file=args.results_file)
    elif args.dataset == "video_mme_v2":
        dataset = Video_MME_v2(dataset_dir=args.dataset_dir, results_file=args.results_file)

    print(f"[INFO] Total questions: {len(dataset.data)}, Unprocessed questions: {len(dataset.unprocessed_data)}")

    if len(dataset.unprocessed_data) > 0:
        asyncio.run(main(args, dataset))
    else:
        dataset.calculate_accuracy()
