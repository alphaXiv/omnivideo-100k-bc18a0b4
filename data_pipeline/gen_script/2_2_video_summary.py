import os
import time
import json
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
Generate a brief, high-level summary of the provided video. The description must be based only information from both the video's visuals (what is seen) and its audio (what is heard).

# Constraints
1. You MUST NOT include any speculation.
2. You must use the exact name from the "Main Entities List" when referring to any listed entity.
Main Entities List:
{MAIN_ENTITIES}

# Output Format
Directly provide the factual summary in a single, concise paragraph.
""".strip()
# MAIN_ENTITIES: - [entity]: [description]


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
    args.main_entities_file = os.path.join(args.script_folder, "1_1_main_entities.jsonl")
    args.video_summary_file = os.path.join(args.script_folder, "2_2_video_summary.jsonl")
    return args


async def call_api(client, video_bytes, prompt, timeout):
    def sync_call():
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
                video_bytes = f.read()
        except:
            print(f"[ERROR] Cannot read video: {video_path}")
            return None

        main_entites_str = ""
        for entity in item["main_entities"]:
            main_entites_str += f"- {entity["entity"]}: {entity["description"]}\n"

        for client in clients:
            try:
                run_start = time.time()

                response = await call_api(client, video_bytes,
                                        prompt.format(MAIN_ENTITIES=main_entites_str.strip()),
                                        timeout=TIMEOUT_LIMIT)
                if not response:
                    print("[TIMEOUT] API call exceeded timeout:", TIMEOUT_LIMIT)
                    continue

                item["video_summary"] = response.text
                
                run_end = time.time()
                tokens_data = response.usage_metadata
        
                item.setdefault("run_data", {})
                item["run_data"]["video_summary"] = {
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

    data = {}
    with open(args.main_entities_file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            item = json.loads(line)
            data[item["id"]] = item

    if os.path.exists(args.video_summary_file):
        with open(args.video_summary_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                item = json.loads(line)
                data[item["id"]] = item

    unprocessed_data = []
    for item in data.values():
        if "video_summary" in item and item["video_summary"]:
            continue
        unprocessed_data.append(item)

    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    file_lock = asyncio.Lock()
    async with aiofiles.open(args.video_summary_file, "a", encoding="utf-8") as f:
        tasks = []
        for item in unprocessed_data:
            tasks.append(asyncio.create_task(process_single_video_concurrent(item, args.root_path, clients, semaphore, file_lock, f)))

        if tasks:
            for f_task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing Videos"):
                await f_task
        
    os._exit(0)


if __name__ == "__main__":
    asyncio.run(main())
