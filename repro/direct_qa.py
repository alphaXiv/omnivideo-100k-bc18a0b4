"""Direct single-pass QA generation baseline for the ablation in the paper
(Section 5.4, "Clue-Guided QA Generation": clue-guided vs direct generation).

The released engine generates cross-segment QA in two steps: global clue mining
(prompt_1, which emits "Relevant Segments" timestamps) then locally-focused
generation (prompt_2). This baseline collapses that into ONE pass over the same
structured script: it asks the model to produce a challenging audio-visual QA for
the task and to report which segment time-ranges it used. Measuring the temporal
span of the cited segments (metrics.py) reproduces the paper's claim that
clue-guided generation yields longer-temporal-span, more cross-segment QA than
direct generation.

The script format fed to the model is byte-identical to the engine's own
`get_segments_description`, so the only variable is one-step vs two-step."""

import os
import json
import argparse
from google import genai

API_KEY = os.environ["API_KEY"]
MODEL_NAME = os.environ["MODEL_NAME"]

DIRECT_PROMPT = """
# Role
You are a multimodal video QA generator.

# Task
You are given a structured textual description of a video: a summary, a main
entity list, and a timestamped segment-by-segment script. In a single step,
directly generate one "{TASK}" question and its answer from this script.

# Input
Video Summary:
{VIDEO_SUMMARY}
Main Entities:
{MAIN_ENTITIES}
Detailed Script:
{SEGMENTS}

# Output Format (strict)
Q: [the question, in natural language, no timestamps]
A: [the answer, in natural language, no timestamps]
USED_SEGMENTS: [list at most two segment time ranges that contain the actual
evidence strictly necessary to answer the question, each exactly as
"MM:SS-MM:SS", comma-separated. Do NOT list segments you merely consulted for
context, background, or disambiguation; only the 1-2 segments whose content the
answer directly depends on.]
""".strip()


def mmss_to_seconds(mmss):
    mm, ss = mmss.split(":")
    return int(mm) * 60 + int(ss)


def get_segments_description(item):
    s = ""
    for seg in item["segments"]:
        s += f"[{seg['start_time']} - {seg['end_time']}]\n"
        audio = sorted(seg["transcription"] + seg["non_speech"],
                       key=lambda x: (mmss_to_seconds(x["start_time"]), mmss_to_seconds(x["end_time"])))
        astr = ""
        for a in audio:
            if "text" in a:
                astr += f"({a['start_time']}-{a['end_time']}) [{a['speaker']}]: {a['text']}\n"
            if "sound" in a:
                astr += f"({a['start_time']}-{a['end_time']}) ({a['sound']})\n"
        if astr:
            s += f"AUDIO:\n{astr}"
        vstr = ""
        if len(seg["visual"]) == 1:
            for v in seg["visual"]:
                if "text" in v:
                    vstr += f"{v['text'].replace(chr(10) + chr(10), chr(10))}\n"
        else:
            for v in seg["visual"]:
                if "text" in v:
                    vstr += f"({v['start_time']}-{v['end_time']})\n{v['text'].replace(chr(10) + chr(10), chr(10))}\n"
                else:
                    vstr += f"({v['start_time']}-{v['end_time']})\n"
        if vstr:
            s += f"VISUAL:\n{vstr}"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_path", required=True)
    ap.add_argument("--task", required=True)
    args = ap.parse_args()

    script_file = os.path.join(args.root_path, "script.jsonl")
    out_dir = os.path.join(args.root_path, "qa_files")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"direct_{args.task}.jsonl")

    client = genai.Client(api_key=API_KEY)
    task_pretty = args.task.replace("_", " ")

    items = []
    with open(script_file, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))

    with open(out_file, "w", encoding="utf-8") as f:
        for item in items:
            ents = "".join(f"- {e['entity']}: {e['description']}\n" for e in item["main_entities"])
            segs = get_segments_description(item)
            prompt = DIRECT_PROMPT.format(TASK=task_pretty, VIDEO_SUMMARY=item["video_summary"].strip(),
                                          MAIN_ENTITIES=ents.strip(), SEGMENTS=segs.strip())
            try:
                resp = client.models.generate_content(model=MODEL_NAME, contents=prompt)
                text = resp.text
            except Exception as e:  # noqa: BLE001
                print(f"[FAILED] {item['id']} {args.task}: {e}")
                continue
            f.write(json.dumps({"id": item["id"], "task": args.task, "direct_text": text}) + "\n")
            f.flush()
            print(f"[SUCESS] direct {args.task} {item['id']}")


if __name__ == "__main__":
    main()
