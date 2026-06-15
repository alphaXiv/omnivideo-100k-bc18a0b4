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
You are an AI Acoustic Event Logger.

# Task
Your task is to identify all sound events, excluding discernible dialogue, and generate a strictly objective, chronological log. The output MUST be either a single, valid JSON array of objects or the literal value 'None'.

# Constraints
1. You must only describe the sound itself using general, objective categories. Strictly avoid inferring the source, cause, intent, or context. Specially, If a sound is identified as music, you must provide a more detailed description.
- Correct (Objective Description): "Glass breaking sound", "Footsteps", "Metallic impact", "Dog bark", "Laughter", "Heavy breathing sound"
- Incorrect (Inferred Context): "A window shattered", "Someone is running", "A wrench was dropped", "He is scared"
2. Do not log human dialogue. However, non-dialogue human sounds (like a laugh, cough, cry) should be logged.
3. Timestamps should mark the beginning and the end of the sound. The string format MUST be strictly "MM:SS" (minutes:seconds).
4. If no non-dialogue sound events are detected in the entire video, your output MUST be the single literal value 'None', without any JSON formatting or quotes.

# Output Format (if sounds are detected)
[
  {
    "start_time": "00:15",
    "end_time": "00:16",
    "sound": "Metallic impact"
  }
]

# Output Format (if no sounds are detected)
None
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
    args.non_speech_file = os.path.join(args.script_folder, "1_2_non_speech.jsonl")
    return args


async def call_api(client, audio_bytes, audio_type, prompt, timeout):
    def sync_call():
        return client.models.generate_content(
            model=MODEL_NAME,
            contents=types.Content(
                parts=[
                    types.Part.from_bytes(data=audio_bytes, mime_type=f"audio/{audio_type}"),
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
    audio_path = os.path.join(root_path, item["a_path"])

    async with semaphore:
        try:
            with open(audio_path, "rb") as f:
                audio_bytes = await asyncio.to_thread(f.read)
        except:
            print(f"[ERROR] Cannot read video: {audio_path}")
            return None

        for client in clients:
            try:
                run_start = time.time()

                response = await call_api(client, audio_bytes, os.path.splitext(audio_path)[1].lstrip("."),
                                          prompt, timeout=TIMEOUT_LIMIT)
                if not response:
                    print("[TIMEOUT] API call exceeded timeout:", TIMEOUT_LIMIT)
                    continue

                non_speech_res = response.text
                if non_speech_res.startswith("```json"):
                    non_speech_res = non_speech_res.removeprefix("```json")
                if non_speech_res.endswith("```"):
                    non_speech_res = non_speech_res.removesuffix("```")
                if non_speech_res != "None":
                    item["non_speech"] = json.loads(non_speech_res)
                else:
                    item["non_speech"] = None
                # print(item["non_speech"])

                run_end = time.time()
                tokens_data = response.usage_metadata

                item.setdefault("run_data", {})
                item["run_data"]["non_speech"] = {
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

    if os.path.exists(args.non_speech_file):
        with open(args.non_speech_file, "r", encoding="utf-8") as f:
            data_tmp += [json.loads(line) for line in f.readlines()]

    unprocessed_data = []
    data = {item["id"]: item for item in data_tmp}
    for item in data.values():
        if "non_speech" in item:
            continue
        unprocessed_data.append(item)

    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    file_lock = asyncio.Lock()
    async with aiofiles.open(args.non_speech_file, "a", encoding="utf-8") as f:
        tasks = []
        for item in unprocessed_data:
            tasks.append(asyncio.create_task(process_single_video_concurrent(item, args.root_path, clients, semaphore, file_lock, f)))

        if tasks:
            for f_task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing Videos"):
                await f_task

    os._exit(0)


if __name__ == "__main__":
    asyncio.run(main())
