import os
import time
import json
import math
import asyncio
import aiofiles
import argparse
from tqdm import tqdm
from google import genai
from google.genai import types


prompt = """
# Role
You are an AI expert in video content analysis.

# Your Task
Identify the main active entities (people, animals and objects) in the provided video and return the findings as a single, clean, and strictly valid JSON array.

# Instructions
Follow these steps meticulously:
1. Process the entire video, including audio, to grasp its core content.
2. Strict Entity Filtering:
- Your list must only include entities that are critical to the video's narrative or primary action.
- You must ignore all minor elements, such as static backgrounds or irrelevant passersby.
3. Data Extraction for Each Entity:
- entity: Assign a descriptive but unique name.
- description: Write a brief, factual, and distinguishing description.
- label: Assign exactly one of the following lowercase strings: people, animal, object.
4. JSON Formatting: Your response must be a single, valid JSON array. Do not include any explanatory text, comments, or markdown formatting.

# Output Format
[
  {
    "entity": "<name>",
    "description": "<clear and factual description>",
    "label": "<people/animal/object>"
  }
]
""".strip()


API_KEY = os.environ["API_KEY"]
MODEL_NAME = os.environ["MODEL_NAME"]
TIMEOUT_LIMIT = int(os.environ.get("TIMEOUT_LIMIT", 300))
CONCURRENCY_LIMIT = int(os.environ.get("CONCURRENCY_LIMIT", 50))
BASEURL_POOL = os.environ.get("BASEURL_POOL", None).split(",")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, required=True)

    args = parser.parse_args()

    args.script_folder = os.path.join(args.root_path, "script_files")
    args.videos_list_file = os.path.join(args.script_folder, "0_videos_sep_low.jsonl")
    args.main_entities_file = os.path.join(args.script_folder, "1_1_main_entities.jsonl")
    return args


def estimate_payload_size(video_bytes, prompt_text):
    video_raw_size = len(video_bytes)
    video_b64_size = math.ceil(video_raw_size / 3) * 4

    text_size = len(prompt_text.encode("utf-8"))

    json_overhead = 1024

    total_bytes = video_b64_size + text_size + json_overhead
    total_mb = total_bytes / (1024 * 1024)

    return total_mb


async def call_api(client, video_bytes, prompt, timeout):
    def sync_call():
        # print(estimate_payload_size(video_bytes, prompt))

        return client.models.generate_content(
            model=MODEL_NAME,
            contents=types.Content(
                parts=[
                    types.Part(inline_data=types.Blob(data=video_bytes, mime_type='video/mp4')),
                    types.Part(text=prompt)
                ]
            )
        )
    
    try:
        response = await asyncio.wait_for(asyncio.to_thread(sync_call), timeout=timeout)
        return response
    except asyncio.TimeoutError:
        return None


async def process_single_video_concurrent(item, root_path, clients, semaphore, file_lock, file_handle):
    video_id = item["id"]
    video_path = os.path.join(root_path, item["low_path"])

    async with semaphore:
        try:
            with open(video_path, "rb") as f:
                video_bytes = await asyncio.to_thread(f.read)
        except:
            print(f"[ERROR] Cannot read video: {video_path}")
            return

        for client in clients:
            try:
                run_start = time.time()

                response = await call_api(client, video_bytes, prompt, timeout=TIMEOUT_LIMIT)
                if not response:
                    print("[TIMEOUT] API call exceeded timeout:", TIMEOUT_LIMIT)
                    continue

                main_entities_res = response.text
                if main_entities_res.startswith("```json"):
                    main_entities_res = main_entities_res.removeprefix("```json")
                if main_entities_res.endswith("```"):
                    main_entities_res = main_entities_res.removesuffix("```")
                item["main_entities"] = json.loads(main_entities_res)
                # print(item["main_entities"])

                run_end = time.time()
                tokens_data = response.usage_metadata
                item.setdefault("run_data", {})
                item["run_data"]["main_entities"] = {
                    "input": tokens_data.prompt_token_count,
                    "output": tokens_data.candidates_token_count,
                    "thinking": tokens_data.thoughts_token_count,
                    "cost_time": run_end - run_start
                }

                async with file_lock:
                    await file_handle.write(json.dumps(item) + "\n")
                    await file_handle.flush()
                print("[SUCESS]", video_id)
                return

            except Exception as e:
                print(f"[Retry]: Error on {video_id} {e}")

        print(f"[FAILED]: {video_id} failed after {len(clients)} retries")
        return


async def main():
    args = get_args()
    if BASEURL_POOL:
        clients = [genai.Client(
            api_key=API_KEY,
            http_options={
                "base_url": base_url
            }
        ) for base_url in BASEURL_POOL]
    else:
        clients = [genai.Client(api_key=API_KEY)]

    with open(args.videos_list_file, "r") as f:
        data_tmp = [json.loads(line) for line in f.readlines()]

    if os.path.exists(args.main_entities_file):
        with open(args.main_entities_file, "r", encoding="utf-8") as f:
            data_tmp += [json.loads(line) for line in f.readlines()]

    unprocessed_data = []
    data = {item["id"]: item for item in data_tmp}
    for item in data.values():
        if "main_entities" in item and item["main_entities"]:
            continue
        unprocessed_data.append(item)

    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    file_lock = asyncio.Lock()
    async with aiofiles.open(args.main_entities_file, "a", encoding="utf-8") as f:
        tasks = []
        for item in unprocessed_data:
            tasks.append(asyncio.create_task(process_single_video_concurrent(item, args.root_path, clients, semaphore, file_lock, f)))

        if tasks:
            for f_task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing Videos"):
                await f_task

    os._exit(0)


if __name__ == "__main__":
    asyncio.run(main())
