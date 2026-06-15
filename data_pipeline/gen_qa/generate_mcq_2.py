import os
import re
import time
import json
import random
import asyncio
import argparse
import aiofiles
from tqdm import tqdm
from google import genai


comparison_prompt_1 = """
# Role
You are an expert Video Content Analyst, specializing in Differentiation and Nuanced Analysis. Your expertise lies in distinguishing between distinct but related concepts, actions, states or events by synthesizing audio and visual cues.

# Task
Analyze the provided detailed video description to exhaustively uncover nuanced distinctions. You will do this by identifying groups of video segments where two entities or events are conceptually related but differ in specific aspect. You must demonstrate exactly what those differences are by synthesizing audio and visual cues.

# Input
You will be provided with the following information derived from a video:
Video Summary:
{VIDEO_SUMMARY}
Main Entities:
{MAIN_ENTITIES}
SEGMENTS:
{SEGMENTS}

# Instructions
1. Identify Related but Distinct Subjects:
Meticulously scan the entire video description for pairs of events or entities that share a common theme but represent different variations. Avoid obvious, unrelated opposites (like "Cat vs. Car").
2. Pinpoint the Nuance:
Determine exactly what the subtle differentiator is.
3. Group and Link:
List the relevant segments. You must ensure that the comparison relies on combining audio descriptions from one part with visual evidence from another.

# Constraints
1. Your analysis must be exhaustive. You are required to identify all distinct, well-supported comparative distinctions capable of being inferred from the video..
2. Each group you identify must include at least two non-consecutive video segments.
3. If you cannot identify any groups of segments that meet all the above criteria after a thorough analysis, your entire response must be the single word NONE. Do not provide any other text or explanation.

# Output Format
Format your response precisely as follows for each group you identify:
Associated Group [Number]: [A concise title describing the group's central theme]
Relevant Segments:
[Timestamp of the first relevant segment]
[Timestamp of the second relevant segment]
[Add all other relevant timestamps as needed]
Analysis of Connection:
[Detail precisely how audio clues from one or more segments and visual evidence from another one or more segments interact to help understanding the nuance.]
""".strip()


sentiment_analysis_prompt_1 = """
# Role
You are an expert Video Content Analyst, specializing in Sentiment Analysis. Your expertise lies in decoding complex human internal states by synthesizing disparate elements-dialogue, sounds, actions, and visual cues-from across an entire video.

# Task
Analyze the provided detailed video description to exhaustively uncover hidden or complex instances of specific attitudes, emotions, tones, or character traits. You will do this by identifying groups of interconnected video segments that collectively establish how a character feels or thinks about a specific subject. You will demonstrate that the true nature of the sentiment or character trait is only understood by linking the audio of one segment with the visuals of another (or multiple others).

# Input
You will be provided with the following information derived from a video:
Video Summary:
{VIDEO_SUMMARY}
Main Entities:
{MAIN_ENTITIES}
SEGMENTS:
{SEGMENTS}

# Instructions
1. Identify All Hidden Sentiment Chains:
Meticulously scan the entire video description to pinpoint all moments where a character's internal state (attitude, emotion, tone) is not immediately obvious from a single moment. Focus on sentiments that are revealed non-linearly.
2. Group Contextually-Linked Segments:
For each sentiment chain you identify, find and list all relevant video segments that, when combined, comprehensively present the character's specific attitude, emotion, or tone. These segments, viewed together, should build a complete picture of the character's state.
3. Analyze the Cross-Modal Synthesis:
For each group, your analysis must prove that the specific attitude/emotion is only correctly defined when audio elements from one or more segments are combined with visual elements from another one or more segments.

# Constraints
1. Your analysis must be exhaustive. You are required to identify all distinct groups that meet the criteria within the entire video, ensuring that every significant sentiment chain is accounted for.
2. Each group you identify must include at least two non-consecutive video segments.
3. If you cannot identify any groups of segments that meet all the above criteria after a thorough analysis, your entire response must be the single word NONE. Do not provide any other text or explanation.

# Output Format
Format your response precisely as follows for each group you identify:
Associated Group [Number]: [A concise title describing the group's central theme]
Relevant Segments:
[Timestamp of the first relevant segment]
[Timestamp of the second relevant segment]
[Add all other relevant timestamps as needed]
Analysis of Connection:
[Detail precisely how audio elements from one or more segments and visual elements from another one or more segments interact to reveal the complete nature of the sentiment/attitude.]
""".strip()


summarization_prompt_1 = """
# Role
You are an expert Video Analyst specializing in Topical Cohesion. Your expertise lies in identifying and grouping disparate video segments that, when combined, collectively explain a specific process, describe a multi-part event, or develop a central topic.

# Task
Analyze the provided detailed video description to identify all groups of interconnected segments that collectively cover a single, coherent topic. For each topic, you will group the relevant segments and explain how they build a complete picture through a combination of audio and visual information.

# Input
You will be provided with the following information derived from a video:
Video Summary:
{VIDEO_SUMMARY}
Main Entities:
{MAIN_ENTITIES}
SEGMENTS:
{SEGMENTS}

# Instructions
1. Identify Coherent Topics:
Meticulously scan the entire video description to identify significant topics that are explained or developed across multiple segments. These can be abstract themes (e.g., "a character's guilt") or concrete processes/events (e.g., "the arguments for a new plan").
2. Group All Supporting Segments:
For each distinct topic identified, find and list all relevant video segments that contribute to its explanation or development. These segments, viewed together, should provide a comprehensive understanding of the topic.
3. Analyze the Cross-Modal Development:
For each group, your analysis must explain how the topic is developed through a synthesis of audio and visual elements across the different segments. You must explain how combining audio elements from one or more segments with visual elements from another one or more segments creates a richer understanding of the topic.

# Constraints
1. Your analysis must be exhaustive. You are required to identify all distinct thematic groups that meet the criteria within the entire video.
2. Each group you identify must include at least two non-consecutive video segments.
3. If you cannot identify any groups of segments that meet all the above criteria after a thorough analysis, your entire response must be the single word NONE. Do not provide any other text or explanation.

# Output Format
Format your response precisely as follows for each group you identify:
Associated Group [Number]: [A concise title describing the group's central theme]
Relevant Segments:
[Timestamp of the first relevant segment]
[Timestamp of the second relevant segment]
[Add all other relevant timestamps as needed]
Analysis of Connection:
[Detail precisely how audio clues and visual evidence from the various segments interact to provide a complete explanation or overview of the topic.]
""".strip()


causal_reasoning_prompt_1 = """
# Role
You are an expert Video Content Analyst, specializing in Causal Chain Analysis. Your expertise lies in uncovering the root causes of events by synthesizing disparate elements-dialogue, sounds, actions, and visual cues-from across an entire video.

# Task
Analyze the provided detailed video description to exhaustively uncover all hidden or complex causal relationships. You will do this by identifying and explaining groups of interconnected video segments that collectively establish a clear cause-and-effect chain. You will demonstrate that the true reason for a key event or outcome is only understood by linking the audio of one segment with the visuals of another (or multiple others).

# Input
You will be provided with the following information derived from a video:
Omni-modal Captions:
{SEGMENTS}

# Instructions
1. Identify All Hidden Causal Chains:
Meticulously scan the entire video description to pinpoint all events or outcomes whose root causes are not immediately apparent. Focus on cause-and-effect relationships that are revealed non-linearly or implicitly.
2. Group All Causally-Linked Segments:
For each causal chain you identify, find and list all relevant video segments that, when combined, connect a cause to its effect. These segments, viewed together, should build a complete picture of why a specific outcome occurred.
3. Analyze the Cross-Modal Synthesis:
For each group, your analysis must prove that the causal link is only correctly and fully understood when audio elements from one or more segments are combined with visual elements from another one or more segments.

# Constraints
1. Your analysis must be exhaustive. You are required to identify all distinct groups that meet the criteria within the entire video, ensuring that every significant causal chain is accounted for.
2. Each group you identify must include at least two non-consecutive video segments.

# Output Format
Format your response precisely as follows for each group you identify:
Associated Group [Number]: [A concise title describing the group's central theme]
Relevant Segments:
[Timestamp of the first relevant segment]
[Timestamp of the second relevant segment]
[Add all other relevant timestamps as needed]
Analysis of Connection:
[Detail precisely how audio elements from one or more segments and visual elements from another one or more segments interact to reveal the complete causal chain.]
""".strip()


future_prediction_prompt_1 = """
# Role
You are an expert Video Content Analyst, specializing in Predictive Narrative Analysis. Your expertise lies in forecasting future events that are likely to occur after the conclusion of a video by synthesizing disparate audio and visual clues from across the entire narrative.

# Task
Analyze the provided detailed video description to forecast plausible future events. You will do this by identifying and explaining groups of interconnected video segments that collectively provide a strong evidentiary basis for a specific prediction. You will demonstrate that the prediction is not mere speculation, but a logical extrapolation based on the audio and visual clues presented within the video.

# Input
You will be provided with the following information derived from a video:
Omni-modal Captions:
{SEGMENTS}

# Instructions
1. Identify All Predictive Evidence:
Meticulously scan the entire video description to pinpoint all unresolved plot points, stated but unfulfilled intentions, established character behavior patterns, and lingering conflicts.
2. Group All Supporting Segments:
For each plausible future event you predict, find and list all relevant video segments that, when combined, form the logical foundation for that prediction. These segments, viewed together, should build a compelling case for the forecasted outcome.
3. Analyze the Cross-Modal Synthesis:
For each group, your analysis must prove that the prediction is a well-supported inference derived specifically from a cross-modal synthesis. You must explain how combining audio elements from one or more segments with visual elements from another one or more segments logically points to a specific event that is likely to occur after the video's conclusion.

# Constraints
1. Your analysis must be exhaustive. You are required to identify all distinct, well-supported predictions that can be made from the evidence within the entire video.
2. Each group you identify must include at least two non-consecutive video segments.

# Output Format
Format your response precisely as follows for each group you identify:
Associated Group [Number]: [A concise title describing the group's central theme]
Relevant Segments:
[Timestamp of the first relevant segment]
[Timestamp of the second relevant segment]
[Add all other relevant timestamps as needed]
Analysis of Connection:
[Detail precisely how audio clues from one or more segments and visual evidence from another one or more segments interact to build a compelling, logical case for the predicted future event.]
""".strip()


hypothetical_reasoning_prompt_1 = """
# Role
You are an expert Video Content Analyst, specializing in Causal Inference and Counterfactual Simulation. Your expertise lies in constructing "what-if" scenarios by synthesizing interdependent audio and visual clues.

# Task
Analyze the provided detailed video description to construct plausible hypothetical reasoning scenarios. You will do this by identifying specific events or conditions that, if changed (the Hypothesis), would lead to a different, predictable outcome (the Consequence). The logic must require combining audio evidence from one or more segments with visual evidence from another. If the reasoning works using only visual or only audio, it is not a valid output.

# Input
You will be provided with the following information derived from a video:
Video Summary:
{VIDEO_SUMMARY}
Main Entities:
{MAIN_ENTITIES}
SEGMENTS:
{SEGMENTS}

# Instructions
1. Identify Cross-Modal Divergence Points:
Meticulously scan the entire video description to pinpoint moments where the outcome was determined by the interplay of audio and visual factors. Ask: "If this audio/visual condition were different, waht would happen?"
2. Group All Supporting Segments:
For each hypothetical scenario, find and list all relevant video segments. You need segments that establish the original context and segments that provide the evidence for the new outcome.
3. Analyze the Cross-Modal Synthesis:
For each group, your analysis must explain how combining audio elements and visual elements from different segments proves that your hypothetical outcome is grounded in the video's internal logic, not random guessing.

# Constraints
1. Your analysis must be exhaustive. You are required to identify all distinct, well-supported hypothetical scenarios capable of being inferred from the video.
2. Each group you identify must include at least two non-consecutive video segments.
3. If you cannot identify any groups of segments that meet all the above criteria after a thorough analysis, your entire response must be the single word NONE. Do not provide any other text or explanation.

# Output Format
Format your response precisely as follows for each group you identify:
Associated Group [Number]: [A concise title describing the group's central theme]
Relevant Segments:
[Timestamp of the first relevant segment]
[Timestamp of the second relevant segment]
[Add all other relevant timestamps as needed]
Analysis of Connection:
[Detail precisely how audio clues from one or more segments and visual evidence from another one or more segments interact to prove that if the hypothesis occurred, the reasoned consequence would inevitably follow based on the video's logic.]
""".strip()


cross_segments_mcq_prompt = """
# Role
You are an expert specializing in Multimodal Video Understanding, specializing in constructing challenging multiple-choice questions (MCQs) that strictly test a user's understanding ability.

# Task
Analyze the provided textual description of a video (detailed script, a designated set of segments, and auxiliary analysis) to generate a high-difficulty {TASK_TYPE} Multiple Choice Question (MCQ).
{TASK_TYPE} Definition: {TASK_DEFINITION}
Example: {EXAMPLE}

# Input
Detailed Script:
{SEGMENTS}
Designated Segments:
{DESIGNATED_SEGMENTS}
Connections
{CONNECTIONS}

# Instructions
1. Analyze & Locate:
Examine the Designated Segments, referring to the provided connections, to identify the specific Audio cues and Visual cues.
Identify the Core Conclusion that satisfies the {TASK_TYPE} from these cues. 
2. Construct Question:
Based on the Core Conclusion, write a precise question {TASK_DEFINITION}.
Do NOT include phrases like "Based on the audio...", "Considering the visual context...", or "In the final scene...".
Do NOT refer to any "provided analysis", "evidence", or the process of deduction. It must be a direct question about the events in the video.
3. Generate Concise Options:
Create 1 Correct Option (A) and 3 Hard Distractors (B,C,D) for your new question. Options must contain **ONLY the answer label**. Do **NOT** explain the reason or cite evidence within the option text.
To ensure the MCQ is challenging, you must avoid random or obviously false distractors. You can refer to the following strategies to create "Hard Negatives":
- The Plausible Inference Trap (Over-Interpretation):
Create an option that logically fits the scene context and seems reasonable, but is **not explicitly supported** by the evidence.
- Missing Modality (The Half-Blind):
Construct an option that would be correct if you *only* relied on the Visuals (ignoring Audio) or *only* relied on the Audio (ignoring Visuals).
- The Semantic Drift (Nuance Distortion):
Use a label that is closely related to the truth but differs in **Intensity**, **Duration** (State vs Trait), or **Social Nuance** (e.g., "Arrogance" vs "Confidence").
4. Validation & Iteration Loop (Strict Enforcement):
Before producing the final output, perform the following "Solvability Tests" on your draft. If the draft fails ANY check, you **MUST regenerate** the question and options.
- Check 1: The Blind Test (Common Sense)
Can the correct answer be guessed solely by reading the Question and Options without seeing the video?
- Check 2: The Simple Match Test
Can the answer be found by simply matching a keyword from the video or a single visual frame?
- Check 3: The Unimodal Trap
Is the answer derivable from Audio alone or Visual alone?

# Constraints
1. Fixed Position: Option A MUST always be the Correct Answer.
2. Uniqueness: Ensure the correct answer is unambiguously correct based on the provided video.
3. Conciseness: Avoid wordy phrases. Options must contain only the necessary differentiating keywords to remain concise.
4. Style & Length Uniformity: All four options (the correct one and the three distractors) must be of similar word count, sentence structure, and complexity.
5. No Timestamps or Time Ranges: Strictly avoid referring to specific timestamps, time ranges, time codes, or segment IDs in either the question or the answer.
6. JSON Formatting: Your response must be a single, valid JSON object. Do not include any explanatory text, comments, or markdown formatting.

# Output Format
{{
    "evidence": "Identify the Visual Clue and the Audio Clue from the script.",
    "question": "Your reformulated question that requires combining these cues.",
    "options": {{
        "A": "Correct Answer",
        "B": "Distractor",
        "C": "Distractor",
        "D": "Distractor"
    }},
    "explanation": "Briefly explain why the correct option is correct, and how the other incorrect options were derived."
}}
""".strip()


API_KEY = os.environ["API_KEY"]
MODEL_NAME = os.environ["MODEL_NAME"]
TIMEOUT_LIMIT = int(os.environ.get("TIMEOUT_LIMIT", 300))
CONCURRENCY_LIMIT = int(os.environ.get("CONCURRENCY_LIMIT", 50))
BASEURL_POOL = os.environ.get("BASEURL_POOL", None).split(",")

qa_num_per_video = int(os.environ.get("QA_NUM", 2))
tasks = ["comparison", "sentiment_analysis", "summarization",
         "causal_reasoning", "future_prediction", "hypothetical_reasoning"]
prompts = {
    "comparison_prompt_1": comparison_prompt_1,
    "comparison_prompt_2": cross_segments_mcq_prompt,
    "sentiment_analysis_prompt_1": sentiment_analysis_prompt_1,
    "sentiment_analysis_prompt_2": cross_segments_mcq_prompt,
    "summarization_prompt_1": summarization_prompt_1,
    "summarization_prompt_2": cross_segments_mcq_prompt,
    "causal_reasoning_prompt_1": causal_reasoning_prompt_1,
    "causal_reasoning_prompt_2": cross_segments_mcq_prompt,
    "future_prediction_prompt_1": future_prediction_prompt_1,
    "future_prediction_prompt_2": cross_segments_mcq_prompt,
    "hypothetical_reasoning_prompt_1": hypothetical_reasoning_prompt_1,
    "hypothetical_reasoning_prompt_2": cross_segments_mcq_prompt
}

task_type = {
    "comparison": "Comparison",
    "sentiment_analysis": "Sentiment Analysis",
    "summarization": "Summarization",
    "causal_reasoning": "Causal Reasoning",
    "future_prediction": "Future Prediction",
    "hypothetical_reasoning": "Hypothetical Reasoning"
}

task_definition = {
    "comparison": "asks about nuanced differences between related entities, events, or states.",
    "sentiment_analysis": "asks about human's complex internal states (emotions, attitudes, tones, or traits).",
    "summarization": "asks for a overview of a topic, process, or event.",
    "causal_reasoning": "asks about the root causes of events or outcomes.",
    "future_prediction": "asks to forecast likely future events occurring after the video concludes.",
    "hypothetical_reasoning": "asks to infer alternative outcomes of 'what-if' scenarios."
}

example = {
    "comparison": """
How does the environment differ when discussing sales versus defending the game?
A. Grassy vs. Rocky terrain
B. Misty vs. Futuristic
C. Menu vs. UI elements
D. Day vs. Night
""",
    "sentiment_analysis": """
- What was the protagonist's tone like when collaborating with others for the first time?
A.an excited tone.
B.a doubtful tone.
C.an awkward tone.
D.a regretful tone.
- Which clone comic has won the first place in the protagonist's mind?
A. The person in blue.
B. The person in yellow.
C. The person in grey.
D. The person in purple.
""",
    "summarization": """
What happened to the team's coach during the game?
A. He was ill and unable to participate in the entire competition.
B. He was sent off by the referee.
C. He was provoked by the opposing players.
D. He was collided with by a player.
""",
    "causal_reasoning": """
- Why did the president think the person in the picture was asking about the temperature?
A. Because the president didn't hear clearly.
B. Because he is a black man.
C. Because the temperature units are very complicated.
D. Because his voice was too low.
- What made Steve say 'our little board operator'?
A. the answer 'lips'
B. BEE-STUNG lips
C. LITTLE MILK DUD COMMENT
D. the answer 'nose'
""",
    "future_prediction": """
What is the most likely to happen next?
A. The conductor begins a speech.
B. The audience continues to applaud.
C. The orchestra begins to perform.
D. The violinist starts a solo performance.
""",
    "hypothetical_reasoning": """
Who would fall into the water if Song Yuqi's seat change request as the team leader was not accepted?
A. Zhang Zhenyuan and Fan Chengcheng.
B. Meng Ziyi and Zhang Zhenyuan.
C. Song Yuqi and Meng Ziyi.
D. Fan Chengcheng and Song Yuqi.
"""
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


async def process_video_cross_1(item, clients, prompt, semaphore, file_lock, file_handle):
    video_id = item["id"]
                            
    async with semaphore:
        main_entites_str = ""
        for entity in item["main_entities"]:
            main_entites_str += f"- {entity["entity"]}: {entity["description"]}\n"

        segments_description = get_segments_description(item)
        
        for client in clients:
            try:
                run_start = time.time()

                response = await call_api(client,
                                          prompt.format(VIDEO_SUMMARY=item["video_summary"].strip(),
                                                        MAIN_ENTITIES=main_entites_str.strip(),
                                                        SEGMENTS=segments_description.strip()),
                                          timeout=TIMEOUT_LIMIT)
                if not response:
                    print("[TIMEOUT] API call exceeded timeout:", TIMEOUT_LIMIT)
                    continue

                item["segment_groups"] = response.text

                if item["segment_groups"] == "NONE":
                    item["qa"] = None
                else:
                    pattern = r"Relevant Segments:\s*(.*?)\s*Analysis of Connection:\s*(.*?)(?=\s*Associated Group|$)"
                    matches = re.findall(pattern, response.text, re.DOTALL)
                    item["qa"] = []
                    for designated_segments, connections in matches:
                        item["qa"].append({"designated_segments": designated_segments.strip(),
                                           "connections": connections.strip()})
                # print(item["qa"])
                
                run_end = time.time()
                tokens_data = response.usage_metadata
                item.setdefault("run_data", {})
                item["run_data"]["segment_groups"] = {
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


async def cross_segment_task_prompt_1(clients, task, script_data, segment_groups_file):
    data = script_data.copy()
    if os.path.exists(segment_groups_file):
        with open(segment_groups_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                item = json.loads(line)
                data[item["id"]] = item
        
    unprocessed_data = []
    for item in data.values():
        if "qa" in item:
            continue
        unprocessed_data.append(item)
    
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    file_lock = asyncio.Lock()
    async with aiofiles.open(segment_groups_file, "a", encoding="utf-8") as f:
        tasks = []
        for item in unprocessed_data:
            tasks.append(asyncio.create_task(process_video_cross_1(item, clients, prompts[f"{task}_prompt_1"], semaphore, file_lock, f)))

        if tasks:
            for f_task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing Videos"):
                await f_task
    
    return


async def process_video_cross_2(item, task, clients, prompt, semaphore, file_lock, file_handle):
    video_id = item["id"]
                            
    async with semaphore:
        main_entites_str = ""
        for entity in item["main_entities"]:
            main_entites_str += f"- {entity["entity"]}: {entity["description"]}\n"

        segments_description = get_segments_description(item)
        
        completed_qa_idx = []
        for qa_idx, i in enumerate(item["mcq"]):
            if "content" in i and i["content"]:
                completed_qa_idx.append(qa_idx)
        unfinished_qa_idx = [idx for idx in range(len(item["mcq"])) if idx not in completed_qa_idx]
        sample_qa_idx = random.sample(unfinished_qa_idx, min(len(unfinished_qa_idx), qa_num_per_video - len(completed_qa_idx)))
        
        for qa_idx in sample_qa_idx:
            i = item["mcq"][qa_idx]

            idx = 0
            while idx < len(clients):
                try:
                    run_start = time.time()

                    response = await call_api(clients[idx],
                                            prompt.format(TASK_TYPE=task_type[task].strip(),
                                                          TASK_DEFINITION=task_definition[task].strip(),
                                                          SEGMENTS=segments_description.strip(),
                                                          EXAMPLE=example[task].strip(),
                                                          DESIGNATED_SEGMENTS=i["designated_segments"].strip(),
                                                          CONNECTIONS=i["connections"].strip()),
                                            timeout=TIMEOUT_LIMIT)
                    if not response:
                        print("[TIMEOUT] API call exceeded timeout:", TIMEOUT_LIMIT)
                        continue

                    mcq_res = response.text
                    if mcq_res.startswith("```json"):
                        mcq_res = mcq_res.removeprefix("```json")
                    if mcq_res.endswith("```"):
                        mcq_res = mcq_res.removesuffix("```")
                    i["content"] = json.loads(mcq_res)
                    # print(i["content"])

                    run_end = time.time()
                    tokens_data = response.usage_metadata
                    item.setdefault("run_data", {})
                    item["run_data"].setdefault("mcq", [])
                    item["run_data"]["mcq"].append({
                        "input": tokens_data.prompt_token_count,
                        "output": tokens_data.candidates_token_count,
                        "thinking": tokens_data.thoughts_token_count,
                        "cost_time": run_end - run_start
                    })

                    async with file_lock:
                        await file_handle.write(json.dumps(item) + "\n")
                        await file_handle.flush()
                    print("[SUCESS]", video_id, f"Group {qa_idx}")
                    break

                except Exception as e:
                    print(f"[Retry]: Error on {video_id} Group {qa_idx} {e}")
                    idx += 1

            if idx == len(clients):
                print(f"[FAILED]: {video_id} Group {qa_idx} failed after {len(clients)} retries")


async def cross_segment_task_prompt_2(clients, task, segment_groups_file, qa_file):
    data = {}
    with open(segment_groups_file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            item = json.loads(line)
            data[item["id"]] = item
    if os.path.exists(qa_file):
        with open(qa_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                item = json.loads(line)
                data[item["id"]] = item

    unprocessed_data = []
    for item in data.values():
        if "mcq" not in item:
            item["mcq"] = item["qa"]
        if not item["mcq"]:
            continue

        generated_count = 0
        for i in item["mcq"]:
            if "content" in i and i["content"]:
                generated_count += 1
        if generated_count >= qa_num_per_video or generated_count == len(item["mcq"]):
            continue
        unprocessed_data.append(item)

    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    file_lock = asyncio.Lock()
    async with aiofiles.open(qa_file, "a", encoding="utf-8") as f:
        tasks = []
        for item in unprocessed_data:
            tasks.append(asyncio.create_task(process_video_cross_2(item, task, clients, prompts[f"{task}_prompt_2"], semaphore, file_lock, f)))

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
        segment_groups_file = os.path.join(args.qa_folder, f"{task}_segment_groups.jsonl")
        await cross_segment_task_prompt_1(clients, task, script_data=script_data, segment_groups_file=segment_groups_file)

        qa_file = os.path.join(args.qa_folder, f"{task}_mcq.jsonl")
        await cross_segment_task_prompt_2(clients, task, segment_groups_file=segment_groups_file, qa_file=qa_file)
        print(f"{task} completed")

    os._exit(0)

if __name__ == "__main__":
    asyncio.run(main())
