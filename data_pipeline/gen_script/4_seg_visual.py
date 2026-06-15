import os
import time
import math
import json
import glob
import asyncio
import aiofiles
import argparse
import subprocess
from tqdm import tqdm
import multiprocessing
from google import genai
from google.genai import types


prompt = """
# Role
You are a professional video content analyst, specializing in creating exceptionally detailed and strictly objective visual descriptions.

# Task
Generate a detailed, objective, and chronologically ordered description of the provided video's visual content. Your analysis must be strictly limited to what is directly observable on screen. You are forbidden from making any assumptions, interpretations, or inferences.

# Instructions
For every new shot or significant scene change, you must internally analyze the visual information by considering these four aspects. You will then synthesize these details into a single descriptive paragraph as described in the Output Format.
- Setting & Environment: Location, objects, props, and lighting.
- Characters & Objects: Positioning, posture, and appearance.
- Actions & Interactions: Individual movements and interactions.
- Cinematography: Shot type, camera angle, and movement.

# Constraints
1. Exhaustive Detail: Capture all available visual information, including textures, colors, and background elements.
2. Prohibition of Inference: You are strictly forbidden from inferring relationships ("friend"), internal states ("he thinks..."), or off-screen events (sounds, dialogue).
3. You must use the exact name from the "Main Entities List" when referring to any listed entity.
Main Entities List:
{MAIN_ENTITIES}

# Output Format
- Your entire output must be a single, cohesive text composed of descriptive paragraphs.
- Each new shot or significant scene change must be described in a new paragraph.
- You MUST NOT use bullet points or any of the analytical headings (e.g., 'Setting & Environment') in your final output.
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
    parser.add_argument("--num_chunks", type=int, default=6)
    
    args = parser.parse_args()

    args.script_folder = os.path.join(args.root_path, "script_files")
    args.inte_seg_file = os.path.join(args.script_folder, "3_inte_seg.jsonl")
    args.temp_folder = os.path.join(args.script_folder, "temp_visual_outputs")
    args.script_file = os.path.join(args.root_path, "script.jsonl")
    return args


def mmss_to_seconds(mmss):
    mm, ss = mmss.split(":")
    return int(mm) * 60 + int(ss)


async def extract_clip_bytes(video_path, start_time, end_time):
    cmd = [
        "ffmpeg",
        "-ss", str(start_time),
        "-i", video_path,
        "-t", str(end_time - start_time),
        "-movflags", "frag_keyframe+empty_moov",
        "-f", "mp4",
        "pipe:1"
    ]
    process = await asyncio.create_subprocess_exec(
        *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        print(f"[Error] FFmpeg: {stderr.decode()}")
        return None
    return stdout


async def call_api(client, video_bytes, prompt, timeout):
    def sync_call():
        return client.models.generate_content(
            model=MODEL_NAME,
            contents=types.Content(
                parts=[
                    types.Part(inline_data=types.Blob(data=video_bytes, mime_type='video/mp4')),
                    types.Part(text=prompt)
                ]
            ),
        )
    
    try:
        response = await asyncio.wait_for(asyncio.to_thread(sync_call), timeout=timeout)
        return response
    except asyncio.TimeoutError:
        return None


async def process_segment(video_id, video_path, seg_idx, v_idx, v, main_entities_str, clients, semaphore, file_lock, file_handle):
    async with semaphore:
        start = mmss_to_seconds(v["start_time"])
        end = mmss_to_seconds(v["end_time"])
        if end == start:
            end += 1
        video_bytes = await extract_clip_bytes(video_path, start, end)
        if not video_bytes:
            print(f"[ERROR] Cannot extract clip for {video_id} Segment {seg_idx} {v_idx}")
            return

        for client in clients:
            try:
                run_start = time.time()

                response = await call_api(client, video_bytes,
                                          prompt.format(MAIN_ENTITIES=main_entities_str.strip()),
                                          timeout=TIMEOUT_LIMIT)

                if not response:
                    print("[TIMEOUT] API call exceeded timeout:", TIMEOUT_LIMIT)
                    continue

                v["text"] = response.text
                # print(v["text"])
                            
                run_end = time.time()
                tokens_data = response.usage_metadata
                
                res_data = {
                    "id": video_id,
                    "seg_idx": seg_idx,
                    "v_idx": v_idx,
                    "text": response.text,
                    "run_data": {
                        "input": tokens_data.prompt_token_count,
                        "output": tokens_data.candidates_token_count,
                        "thinking": tokens_data.thoughts_token_count,
                        "cost_time": run_end - run_start
                    }
                }
   
                async with file_lock:
                    await file_handle.write(json.dumps(res_data) + "\n")
                    await file_handle.flush()
                print("[SUCESS]", video_id, f"Segment {seg_idx} {v_idx}")
                return
            except Exception as e:
                print(f"[Retry]: Error on {video_id} Segment {seg_idx} {v_idx} {e}")

        print(f"[FAILED]: {video_id} Segment {seg_idx} {v_idx} failed after {len(clients)} retries")


async def worker_process(chunk_id, data_chunk, temp_output_file):
    if BASEURL_POOL:
        clients = [genai.Client(
            api_key=API_KEY,
            http_options={
                "base_url": base_url
            }
        ) for base_url in BASEURL_POOL]
    else:
        clients = [genai.Client(api_key=API_KEY)]

    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    file_lock = asyncio.Lock()

    async with aiofiles.open(temp_output_file, "a", encoding="utf-8") as f:
        tasks = []
        for data in data_chunk:
            tasks.append(process_segment(data[0], data[1],
                                         data[2], data[3], data[4],
                                         data[5],
                                         clients, semaphore, file_lock, f))

        for f_task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"chunk {chunk_id}", position=chunk_id):
            await f_task
    os._exit(0)


def worker_process_wrapper(chunk_id, data_chunk, temp_output_file):
    if not data_chunk:
        return
    asyncio.run(worker_process(chunk_id, data_chunk, temp_output_file))
        

def main():
    args = get_args()
    os.makedirs(args.temp_folder, exist_ok=True)

    data = {}
    with open(args.inte_seg_file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            item = json.loads(line)
            data[item["id"]] = item

    temp_pattern = os.path.join(args.temp_folder, "temp_*.jsonl")
    for temp_f in glob.glob(temp_pattern):
        with open(temp_f, "r", encoding="utf-8") as f:
            for line in f.readlines():
                log = json.loads(line)
                v_id, s_idx, v_idx = log["id"], log["seg_idx"], log["v_idx"]
                if v_id in data:
                    data[v_id]["segments"][s_idx]["visual"][v_idx]["text"] = log["text"]
                    data[v_id].setdefault("run_data", {})
                    data[v_id]["run_data"].setdefault("visual", [])
                    data[v_id]["run_data"]["visual"].append(log["run_data"])

    processed_ids = []
    if os.path.exists(args.script_file):
        with open(args.script_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                item = json.loads(line)
                processed_ids.append(item["id"])

    unprocessed_data = []
    with open(args.script_file, "a", encoding="utf-8") as f:
        for item in tqdm(data.values()):
            completed = True
            for seg in item["segments"]:
                for v in seg["visual"]:
                    if "text" not in v or not v["text"]:
                        completed = False
                        break
                if not completed:
                    break

            if completed:
                if item["id"] not in processed_ids:
                    f.write(json.dumps(item) + "\n")
                    f.flush()
                    processed_ids.append(item["id"])
            else:
                unprocessed_data.append(item)

    all_tasks = []
    for item in tqdm(unprocessed_data):
        video_path = os.path.join(args.root_path, item["v_path"])
        main_entities_str = ""
        for entity in item["main_entities"]:
            main_entities_str += f"- {entity["entity"]}: {entity["description"]}\n"

        for s_idx, seg in enumerate(item["segments"]):
            for v_idx, v in enumerate(seg["visual"]):
                if "text" not in v or not v["text"]:
                    all_tasks.append((item["id"], video_path,
                                      s_idx, v_idx, v,
                                      main_entities_str))

    if len(all_tasks):
        chunk_size = math.ceil(len(all_tasks) / args.num_chunks)
        chunks = [all_tasks[i: i + chunk_size] for i in range(0, len(all_tasks), chunk_size)]

        processes = []
        for i in range(args.num_chunks):
            chunk = chunks[i]
            temp_file = os.path.join(args.temp_folder, f"temp_{i}.jsonl")

            p = multiprocessing.Process(
                target=worker_process_wrapper,
                args=(i, chunk, temp_file)
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()


if __name__ == "__main__":
    main()
