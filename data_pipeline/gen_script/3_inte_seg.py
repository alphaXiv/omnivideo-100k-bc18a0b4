import os
import json
import argparse
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--max_seg_length", type=int, default=15)

    args = parser.parse_args()

    args.script_folder = os.path.join(args.root_path, "script_files")
    args.non_speech_file = os.path.join(args.script_folder, "1_2_non_speech.jsonl")
    args.label_speaker_file = os.path.join(args.script_folder, "2_1_label_speaker.jsonl")
    args.video_summary_file = os.path.join(args.script_folder, "2_2_video_summary.jsonl")
    args.inte_seg_file = os.path.join(args.script_folder, "3_inte_seg.jsonl")
    return args


def mmss_to_seconds(mmss):
    mm, ss = mmss.split(":")
    return int(mm) * 60 + int(ss)


def seconds_to_mmss(seconds):
    mm, ss = divmod(int(seconds), 60)
    return f"{mm:02d}:{ss:02d}"


def build_segments(asr_segments, duration, max_seg_length):
    # same speaker
    segments = []
    last_asr_end = 0
    for seg in asr_segments:
        start = mmss_to_seconds(seg["start_time"])
        end = mmss_to_seconds(seg["end_time"])

        # non-speech part
        if start > last_asr_end:
            if segments and start - mmss_to_seconds(segments[-1]["start_time"]) < max_seg_length:
                segments[-1]["end_time"] = seg["start_time"]
            else:
                segments.append({"start_time": seconds_to_mmss(last_asr_end),
                                "end_time": seg["start_time"],
                                "transcription": []})

        # speech part
        if segments and start < mmss_to_seconds(segments[-1]["end_time"]):
            segments[-1]["end_time"] = seg["end_time"]
            segments[-1]["transcription"].append({"start_time": seg["start_time"],
                                                  "end_time": seg["end_time"],
                                                  "text": seg["text"],
                                                  "speaker": seg["speaker"]
                                                  })
        elif segments and end - mmss_to_seconds(segments[-1]["start_time"]) < max_seg_length and \
            (not segments[-1]["transcription"] or segments[-1]["transcription"][-1]["speaker"] == seg["speaker"]):
            segments[-1]["end_time"] = seg["end_time"]
            segments[-1]["transcription"].append({"start_time": seg["start_time"],
                                                  "end_time": seg["end_time"],
                                                  "text": seg["text"],
                                                  "speaker": seg["speaker"]
                                                  })
        else:
            segments.append({"start_time": seg["start_time"],
                             "end_time": seg["end_time"],
                             "transcription": [{"start_time": seg["start_time"],
                                                "end_time": seg["end_time"],
                                                "text": seg["text"],
                                                "speaker": seg["speaker"]
                                                }]})
        last_asr_end = end

    # last non-speech part
    if duration > last_asr_end:
        if segments and duration - mmss_to_seconds(segments[-1]["start_time"]) < max_seg_length:
            segments[-1]["end_time"] = seconds_to_mmss(duration)
        else:
            segments.append({"start_time": seconds_to_mmss(last_asr_end),
                             "end_time": seconds_to_mmss(duration),
                             "transcription": []})

    # general
    segments_v2 = []
    for seg in segments:
        start = mmss_to_seconds(seg["start_time"])
        end = mmss_to_seconds(seg["end_time"])

        if segments_v2 and end - mmss_to_seconds(segments_v2[-1]["start_time"]) < max_seg_length:
            segments_v2[-1]["end_time"] = seg["end_time"]
            segments_v2[-1]["transcription"] += seg["transcription"]
        else:
            segments_v2.append(seg)

    # check
    for seg_num in range(1, len(segments_v2)):
        assert mmss_to_seconds(segments_v2[seg_num]["start_time"]) == mmss_to_seconds(segments_v2[seg_num - 1]["end_time"])
    
    for seg in segments_v2:
        seg_len = mmss_to_seconds(seg["end_time"]) - mmss_to_seconds(seg["start_time"])
        seg["visual"] = []

        ex_num = int(seg_len / max_seg_length)
        step = int(seg_len / (ex_num + 1))

        start = mmss_to_seconds(seg["start_time"])
        for _ in range(0, ex_num):
            end = start + step
            seg["visual"].append({"start_time": seconds_to_mmss(start),
                                  "end_time": seconds_to_mmss(end)})
            start = end
        seg["visual"].append({"start_time": seconds_to_mmss(start),
                              "end_time": seg["end_time"]})
    return segments_v2


def combine_non_speech(video_segments, non_speech):
    cut_points = []
    for seg in video_segments[1:]:
        cut_points.append(seg["start_time"])

    sounds = []
    for sound in non_speech:
        start = mmss_to_seconds(sound["start_time"])
        end = mmss_to_seconds(sound["end_time"])

        points_in_seg = [p for p in cut_points if start < mmss_to_seconds(p) < end]
        boundaries = [sound["start_time"]] + points_in_seg + [sound["end_time"]]
        for i in range(len(boundaries) - 1):
            sounds.append({
                "start_time": boundaries[i],
                "end_time": boundaries[i + 1],
                "sound": sound["sound"]
            })

    sounds = sorted(sounds, key=lambda x: (mmss_to_seconds(x["start_time"]), mmss_to_seconds(x["end_time"])))

    for seg in video_segments:
        start = mmss_to_seconds(seg["start_time"])
        end = mmss_to_seconds(seg["end_time"])
        seg["non_speech"] = [s for s in sounds if mmss_to_seconds(s["start_time"]) >= start and mmss_to_seconds(s["end_time"]) <= end]
    return video_segments


if __name__ == "__main__":
    args = get_args()
    
    non_speech = {}
    with open(args.non_speech_file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            item = json.loads(line)
            non_speech[item["id"]] = item

    video_summary = {}
    with open(args.video_summary_file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            item = json.loads(line)
            video_summary[item["id"]] = item
    
    data = {}
    with open(args.label_speaker_file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            item = json.loads(line)
            if item["id"] in non_speech and item["id"] in video_summary:
                item["non_speech"] = non_speech[item["id"]]["non_speech"]
                item["run_data"]["non_speech"] = non_speech[item["id"]]["run_data"]["non_speech"]
                item["video_summary"] = video_summary[item["id"]]["video_summary"]
                item["run_data"]["video_summary"] = video_summary[item["id"]]["run_data"]["video_summary"]
                data[item["id"]] = item

    if os.path.exists(args.inte_seg_file):
        with open(args.inte_seg_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                item = json.loads(line)
                data[item["id"]] = item

    unprocessed_data = []
    for item in data.values():
        if "segments" in item and item["segments"]:
            continue
        unprocessed_data.append(item)

    with open(args.inte_seg_file, "a", encoding="utf-8") as f:
        for item in tqdm(unprocessed_data):
            if not item["video_summary"]:
                continue

            transcribe = item["label_speaker"]
            if transcribe is None:
                transcribe = []

            try:
                transcribe = sorted(transcribe, key=lambda x: (mmss_to_seconds(x["start_time"]), mmss_to_seconds(x["end_time"])))

                video_segments = build_segments(transcribe, duration=item["duration"],
                                                max_seg_length=args.max_seg_length)

                non_speech = item["non_speech"]
                if non_speech is None:
                    non_speech = []

                item["segments"] = combine_non_speech(video_segments, non_speech)
                f.write(json.dumps(item) + "\n")
                f.flush()
            except Exception as e:
                print(f"[ERROR] {item["id"]} {e}")
