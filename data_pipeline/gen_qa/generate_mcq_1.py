import os
import time
import json
import asyncio
import argparse
import aiofiles
from tqdm import tqdm
from google import genai


perception_mcq_prompt = """
# Role
You are an expert in creating challenging multi-modal comprehension assessments, specializing in constructing challenging multiple-choice questions (MCQs) that test a user's ability to precisely correlate audio and visual information in videos.

# Task
You will be provided with:
1. Video (Detailed Script).
2. A Question (Q) derived from a perception task.
3. The Correct Answer (A).
Your goal is to generate a difficult 4-option Multiple Choice Question. The three distinct incorrect options (distractors) must be plausible but clearly wrong based on the video evidence.

# Input
Detailed Script:
{SEGMENTS}
Input Question:
{QUESTION}
Input Correct Answer:
{ANSWER}

# Instructions for Distractor Generation
To ensure the MCQ is challenging, you must avoid random or obviously false distractors. Use the following strategies to create "Hard Negatives":
1. The Temporal Trap (Right Event, Wrong Time):
Create a distractor describing an event that actually happens in the video but occurring at a different time than the specific audio/visual cue mentioned in the question. If the selected event is similar to the correct answer, you must include more specific details that distinguishes it textually.
2. The Entity/Action Swap (Right Time, Wrong Detail):
Create a distractor that focuses on the correct specific moment (matching the cue) but alters the details with another entity or attribute present in the video.
3. The Modality Hallucination (Plausible but False):
Create a distractor that describes an event that fits the general context/scene logically but is factually incorrect according to the video.

# Constraints
1. Style & Length Uniformity: All four options (the correct one and the three distractors) must be of similar word count, sentence structure, and complexity. Do not make the correct answer significantly different from the distractors.
2. Uniqueness: Ensure the correct answer is unambiguously correct based on the provided video.
3. No Timestamps or Time Ranges: Both the question and the answer must be written in a natural, narrative style. Strictly avoid referring to specific timestamps, time ranges, time codes, or segment IDs in either the question or the answer.
4. JSON Formatting: Your response must be a single, valid JSON object. Do not include any explanatory text, comments, or markdown formatting.

# Output Format
{{
  "question": "The input question provided above",
  "correct_option": "The paraphrased correct answer",
  "distractors": {{
    "A": "Option text...",
    "B": "Option text...",
    "C": "Option text...",
  }},
  "explanation": "Briefly explain why the correct option is right and specifically why each distractor is wrong."
}}
""".strip()


context_understanding_mcq_prompt = """
# Role
You are an expert in creating advanced Multi-modal Context Understanding assessments, specializing in constructing challenging multiple-choice questions (MCQs) that test whether a user can correctly identify the causal or descriptive links between Audio and Visual streams.

# Task
You will be provided with:
1. Video (Detailed Script).
2. A Question (Q).
3. The Correct Answer (A).
Your goal is to generate a difficult 4-option Multiple Choice Question. The three distinct incorrect options (distractors) must be plausible but clearly wrong based on the video evidence.

# Input
Detailed Script:
{SEGMENTS}
Input Question:
{QUESTION}
Input Correct Answer:
{ANSWER}

# Instructions for Distractor Generation
To ensure the MCQ is challenging, you must avoid random or obviously false distractors. Use the following strategies to create "Hard Negatives":
1. The Fine-Grained Trap (Visual/Audio Distortion):
Take the Correct Answer and alter a specific meaningful detail with another entity or attribute present in the video.
2. The Co-occurrence Red Herring (Synchronous but Irrelevant):
Create a distractor describing a distinct event or detail that is present in the video simultaneously with the specific cue (audio quote or visual scene) mentioned in the question, but does not answer the specific question asked.
3. The Modality Hallucination (Plausible but False):
Create a distractor describing a logical cause, consequence, or visual detail that aligns with common sense for the given situation but is not explicitly supported by the video.

# Constraints
1. Concisenes: Removing wordy phrases, rewrite the Input Correct Answer and all Distractors to contain only the necessary differentiating keywords. Remove all redundant context establishing the scene.
2. Style & Length Uniformity: All four options (the correct one and the three distractors) must be of similar word count, sentence structure, and complexity. You must pad distractors or condense the correct answer so all 4 options have roughly the same length.
3. Uniqueness: Ensure the correct answer is unambiguously correct based on the provided video.
4. No Timestamps or Time Ranges: Both the question and the answer must be written in a natural, narrative style. Strictly avoid referring to specific timestamps, time ranges, time codes, or segment IDs in either the question or the answer.
5. JSON Formatting: Your response must be a single, valid JSON object. Do not include any explanatory text, comments, or markdown formatting.

# Output Format
{{
  "question": "The input question provided above",
  "correct_option": "The paraphrased correct answer",
  "distractors": {{
    "A": "Option text...",
    "B": "Option text...",
    "C": "Option text...",
  }},
  "explanation": "Briefly explain why the correct option is right and specifically why each distractor is wrong."
}}
""".strip()


event_sequence_ordering_mcq_prompt = """
# Role
You are an expert in Temporal Reasoning and Content Condensation. Your expertise lies in analyzing complex video narratives, distilling key events into concise phrases, and constructing challenging chronological puzzles.

# Task
You will be provided with:
1. Video (Detailed Script).
2. A Question (Q).
3. A list of Event Descriptions. 
4. The correct sequence of letters, e.g., C A B
Your goal is to rewrite each event description into a concise, keyword-focused form and generate three plausible but incorrect sequences to act as "Hard Negative" distractors.

# Input
Detailed Script:
{SEGMENTS}
Input Question:
{QUESTION}
Events to Rewrite:
{EVENTS}
Input Correct Sequence:
{ANSWER}

# Instructions
1. Rewrite Event:
Removing wordy phrases, rewrite each event from the input "Events to Rewrite" to contain only the necessary differentiating keywords.
2. Generate "Hard Negative" Distractors:
Create three distinct and plausible incorrect sequences.

# Constraints
1. Conciseness: The final output must be as short as possible while retaining the event's uniqueness.
2. Factual Accuracy: The rewritten event must accurately represent the core meaning of the original event description without adding new information or changing the facts.
3. Contextual Awareness: Use the provided Detailed Script to understand which details are crucial and which can be omitted.
4. No Timestamps or Time Ranges: Do not include any temporal information in the rewritten event.
5. JSON Formatting: Your response must be a single, valid JSON object. Do not include any explanatory text, comments, or markdown formatting outside of the JSON structure.

# Output Format
{{
    "rewritten_events": {{
        "A": "The first rewritten event ...",
        "B": "The second rewritten event ...",
        "C": "The third rewritten event ...",
        ...
    }},
    "distractors": [
        "The first error sequence, e.g., A B C",
        "The second error sequence, e.g., C B A",
        "The third error sequence, e.g., B A C"
    ]
}}
""".strip()


API_KEY = os.environ["API_KEY"]
MODEL_NAME = os.environ["MODEL_NAME"]
TIMEOUT_LIMIT = int(os.environ.get("TIMEOUT_LIMIT", 300))
CONCURRENCY_LIMIT = int(os.environ.get("CONCURRENCY_LIMIT", 50))
BASEURL_POOL = os.environ.get("BASEURL_POOL", None).split(",")

tasks = ["fine_grained_perception", "scene_transformation_detection", "context_understanding",
         "event_sequence_ordering"]

prompts = {
    "fine_grained_perception": perception_mcq_prompt,
    "scene_transformation_detection": perception_mcq_prompt,
    "context_understanding": context_understanding_mcq_prompt,
    "event_sequence_ordering": event_sequence_ordering_mcq_prompt,
}


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--task", required=True, choices=tasks)

    args = parser.parse_args()
    args.script_file = os.path.join(args.root_path, "script.jsonl")
    args.qa_folder = os.path.join(args.root_path, "qa_files")
    return args


def mmss_to_seconds(mmss):
    mm, ss = mmss.split(":")
    return int(mm) * 60 + int(ss)


def get_segments_description(item):
    segments_description = ""
    for seg in item["segments"]:
        segments_description += f"[{seg["start_time"]} - {seg["end_time"]}]\n"

        audio = sorted(seg["transcription"] + seg["non_speech"], key=lambda x: (mmss_to_seconds(x["start_time"]), mmss_to_seconds(x["end_time"])))
        audio_str = ""
        for a in audio:
            if "text" in a:
                audio_str += f"({a["start_time"]}-{a["end_time"]}) [{a["speaker"]}]: {a["text"]}\n"
            if "sound" in a:
                audio_str += f"({a["start_time"]}-{a["end_time"]}) ({a["sound"]})\n"
        if audio_str != "":
            segments_description += f"AUDIO:\n{audio_str}"
                                    
        visual_str = ""
        if len(seg["visual"]) == 1:
            for v in seg["visual"]:
                if "text" in v:
                    visual_str += f"{v["text"].replace("\n\n", "\n")}\n"
        else:
            for v in seg["visual"]:
                if "text" in v:
                    visual_str += f"({v["start_time"]}-{v["end_time"]})\n{v["text"].replace("\n\n", "\n")}\n"
                else:
                    visual_str += f"({v["start_time"]}-{v["end_time"]})\n"
        if visual_str != "":
            segments_description += f"VISUAL:\n{visual_str}"
    return segments_description


async def call_api(client, prompt, timeout):
    def sync_call():
        return client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )
    
    try:
        response = await asyncio.wait_for(asyncio.to_thread(sync_call), timeout=timeout)
        return response
    except asyncio.TimeoutError:
        return None


async def process_question(item, script, task, clients, semaphore, file_lock, file_handle):
    question_id = item["question_id"]

    async with semaphore:
        main_entites_str = ""
        for entity in script["main_entities"]:
            main_entites_str += f"- {entity["entity"]}: {entity["description"]}\n"

        segments_description = get_segments_description(script)
        
        for client in clients:
            try:
                run_start = time.time()

                if task in ["fine_grained_perception", "scene_transformation_detection", "context_understanding"]:
                    response = await call_api(client,
                                              prompts[task].format(SEGMENTS=segments_description.strip(),
                                                                   QUESTION=item["Q"].strip(),
                                                                   ANSWER=item["A"].strip()),
                                              timeout=TIMEOUT_LIMIT)
                elif task == "event_sequence_ordering":
                    events = ""
                    for no, option in enumerate(item["Options"]):
                        events += f"{chr(65 + no)}. {option}\n"
                    response = await call_api(client,
                                              prompts[task].format(SEGMENTS=segments_description.strip(),
                                                                   QUESTION=item["Q"].strip(),
                                                                   EVENTS=events.strip(),
                                                                   ANSWER=" ".join(item["A"])),
                                              timeout=TIMEOUT_LIMIT)
                if not response:
                    print("[TIMEOUT] API call exceeded timeout:", TIMEOUT_LIMIT)
                    continue

                mcq_res = response.text
                if mcq_res.startswith("```json"):
                    mcq_res = mcq_res.removeprefix("```json")
                if mcq_res.endswith("```"):
                    mcq_res = mcq_res.removesuffix("```")
                item["mcq"] = json.loads(mcq_res)
                # print(item["mcq"])

                run_end = time.time()
                tokens_data = response.usage_metadata
                item.setdefault("run_data", {})
                item["run_data"]["mcq"] = {
                    "input": tokens_data.prompt_token_count,
                    "output": tokens_data.candidates_token_count,
                    "thinking": tokens_data.thoughts_token_count,
                    "cost_time": run_end - run_start
                }

                async with file_lock:
                    await file_handle.write(json.dumps(item) + "\n")
                    await file_handle.flush()
                print("[SUCESS]", question_id)
                return

            except Exception as e:
                print(f"[Retry]: Error on {question_id} {e}")

        print(f"[FAILED]: {question_id} failed after {len(clients)} retries")
        return


async def mcq(clients, task, script_data, oe_file, mcq_file):
    processed_question_id = set()
    if os.path.exists(mcq_file):
        with open(mcq_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                item = json.loads(line)
                processed_question_id.add(item["question_id"])

    unprocessed_data = []
    with open(oe_file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            item = json.loads(line)
            if item["id"] in script_data and item["question_id"] not in processed_question_id:
                unprocessed_data.append(item)
    
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    file_lock = asyncio.Lock()
    async with aiofiles.open(mcq_file, "a", encoding="utf-8") as f:
        tasks = []
        for item in unprocessed_data:
            tasks.append(asyncio.create_task(process_question(item, script_data[item["id"]], task, clients, semaphore, file_lock, f)))

        if tasks:
            for f_task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing Videos"):
                await f_task
    
    return


async def main():
    args = get_args()
    task = args.task
    os.makedirs(args.qa_folder, exist_ok=True)
    if BASEURL_POOL:
        clients = [genai.Client(
            api_key=API_KEY,
            http_options={
                "base_url": base_url
            }
        ) for base_url in BASEURL_POOL]
    else:
        clients = [genai.Client(api_key=API_KEY)]

    script_data = {}
    with open(args.script_file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            item = json.loads(line)
            script_data[item["id"]] = item

    if task in tasks:
        print(f"{task} begin")
        oe_file = os.path.join(args.root_path, f"{task}.jsonl")
        mcq_file = os.path.join(args.qa_folder, f"{task}_mcq.jsonl")
        await mcq(clients, task, script_data=script_data, oe_file=oe_file, mcq_file=mcq_file)
        print(f"{task} completed")

    os._exit(0)


if __name__ == "__main__":
    asyncio.run(main())
