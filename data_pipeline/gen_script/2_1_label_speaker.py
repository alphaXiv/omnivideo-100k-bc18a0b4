import os
import time
import json
import random
import asyncio
import aiofiles
import argparse
from tqdm import tqdm
from google import genai
from google.genai import types


prompt = """
# Role
You are an advanced AI assistant specializing in Audio-Visual Speaker Diarization. Your expertise lies in combining vision and audio analysis to accurately identify speakers.

# Your Task
Your primary task is to process a video and its corresponding JSON transcript. For each segment in the transcript, you must add a new speaker field and assign it the correct speaker's name. This identification MUST be based on a synthesized analysis of both the video's visual and audio track.

# Transcript
{TRANSCRIPT}

# Instructions
1. Focus on the Time Segment:
For each object in the input JSON, precisely locate the segment in the video using its start_time and end_time.
2. Multi-Modal Speaker Identification:
Within each time segment, perform the following analysis:
- Visual Analysis: Observe the video. Whose lip movements are synchronized with the spoken words (text)?
- Audio Analysis: Listen to the voice. Note its unique characteristics (pitch, tone, cadence) to help distinguish between speakers.
- Synthesize and Decide: Combine the visual and auditory evidence. Visual confirmation of synchronized lip movement is the strongest signal for attribution. If the speaker is off-screen, rely solely on voice identification.
3. Create and Assign Speaker Field:
Based on your identification, add a new key-value pair, "speaker": "<name>", to the object.
- If the identified speaker is on the "Main Entities List", you MUST use their exact name.
- If you can consistently identify a speaker who is not on the list, assign a consistent label.
- If visual and audio cues are conflicting or insufficient for a confident identification, you MUST use the string "UNKNOWN".
Main Entities List:
{MAIN_ENTITIES}

# Constraints
1. You MUST NOT modify the start_time, end_time, or text fields.
2. The output MUST be a single, valid JSON array of objects, representing the completed transcript. Do not add any text or explanations outside of the JSON structure.

# Output Format
[
  {{
    "start_time": "00:01",
    "end_time": "00:04",
    "text": "This is the first sentence.",
    "speaker": <name/UNKNOWN>
  }}
]
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
    args.transcribe_file = os.path.join(args.script_folder, "1_3_transcribe.jsonl")
    args.label_speaker_file = os.path.join(args.script_folder, "2_1_label_speaker.jsonl")
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
                
        transcribe_str = json.dumps(item["transcribe"], ensure_ascii=False, indent=2)

        task_clients = list(clients)
        random.shuffle(task_clients)
        for client in task_clients:
            try:
                run_start = time.time()

                response = await call_api(client, video_bytes,
                                        prompt.format(MAIN_ENTITIES=main_entites_str.strip(),
                                                      TRANSCRIPT=transcribe_str.strip()),
                                        timeout=TIMEOUT_LIMIT)

                if not response:
                    print("[TIMEOUT] API call exceeded timeout:", TIMEOUT_LIMIT)
                    continue

                label_speaker_res = response.text
                if label_speaker_res.startswith("```json"):
                    label_speaker_res = label_speaker_res.removeprefix("```json")
                if label_speaker_res.endswith("```"):
                    label_speaker_res = label_speaker_res.removesuffix("```")
                item["label_speaker"] = json.loads(label_speaker_res)
                # print(item["label_speaker"])
                
                run_end = time.time()
                tokens_data = response.usage_metadata
        
                item.setdefault("run_data", {})
                item["run_data"]["label_speaker"] = {
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

    transcribe = {}
    with open(args.transcribe_file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            item = json.loads(line)
            transcribe[item["id"]] = item

    data = {}
    with open(args.main_entities_file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            item = json.loads(line)
            if item["id"] in transcribe:
                item["transcribe"] = transcribe[item["id"]]["transcribe"]
                item["run_data"]["transcribe"] = transcribe[item["id"]]["run_data"]["transcribe"]
                data[item["id"]] = item

    if os.path.exists(args.label_speaker_file):
        with open(args.label_speaker_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                item = json.loads(line)
                data[item["id"]] = item

    unprocessed_data = []
    for item in data.values():
        if "label_speaker" in item:
            continue
        unprocessed_data.append(item)
    
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    file_lock = asyncio.Lock()
    async with aiofiles.open(args.label_speaker_file, "a", encoding="utf-8") as f:
        tasks = []
        for item in unprocessed_data:
            if item["transcribe"] is None:
                item["label_speaker"] = None
                async with file_lock:
                    await f.write(json.dumps(item) + "\n")
                    await f.flush()
                print("[SUCESS]", item["id"])
            else:
                tasks.append(asyncio.create_task(process_single_video_concurrent(item, args.root_path, clients, semaphore, file_lock, f)))
        
        if tasks:
            for f_task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing Videos"):
                await f_task
    
    os._exit(0)
    

if __name__ == "__main__":
    asyncio.run(main())
