import os
import json
import glob
import argparse
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, required=True)
    
    args = parser.parse_args()

    args.script_folder = os.path.join(args.root_path, "script_files")
    args.inte_seg_file = os.path.join(args.script_folder, "3_inte_seg.jsonl")
    args.temp_folder = os.path.join(args.script_folder, "temp_visual_outputs")
    args.script_file = os.path.join(args.root_path, "script.jsonl")
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


if __name__ == "__main__":
    args = get_args()
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

            if completed and item["id"] not in processed_ids:
                f.write(json.dumps(item) + "\n")
                f.flush()
                processed_ids.append(item["id"])

    with open(args.script_file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            item = json.loads(line)
            
            main_entites_str = ""
            for entity in item["main_entities"]:
                main_entites_str += f"- {entity["entity"]}: {entity["description"]}\n"

            segments_description = get_segments_description(item)
            
            VIDEO_SUMMARY = item["video_summary"].strip()
            MAIN_ENTITIES = main_entites_str.strip()
            SEGMENTS = segments_description.strip()
