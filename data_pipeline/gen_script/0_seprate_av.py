import os
import json
import argparse
import subprocess
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--num_processes", type=int, default=4)
    parser.add_argument("--target_fps", type=int, default=1)
    parser.add_argument("--target_max_dim", type=int, default=480)
    
    args = parser.parse_args()

    args.pre_folder = os.path.join(args.root_path, "pre_files")
    args.videos_list_file = os.path.join(args.pre_folder, "final_videos_list.jsonl")
    args.script_folder = os.path.join(args.root_path, "script_files")
    args.sep_videos_list_file = os.path.join(args.script_folder, "0_videos_sep_low.jsonl")
    args.sep_save_folder = os.path.join(args.root_path, "videos/sep")
    args.low_save_folder = os.path.join(args.root_path, "videos/low")
    return args


def process_video(item, args):
    id = item["id"]
    video_file = os.path.join(args.root_path, item["video_path"])

    v_path = os.path.join(args.sep_save_folder, f"{id}.mp4")

    # Audio codec
    try:
        result = subprocess.run([
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_streams", video_file
        ], capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)
        audio_codec = None
        for stream in info["streams"]:
            if stream["codec_type"] == "audio":
                audio_codec = stream["codec_name"]
                break

        if audio_codec == "opus":
            a_path = os.path.join(args.sep_save_folder, f"{id}.opus")
        elif audio_codec == "aac":
            a_path = os.path.join(args.sep_save_folder, f"{id}.aac")
        else:
            print(f"different acodec: {audio_codec} for video {id}")
            return None

        low_path = os.path.join(args.low_save_folder, f"{id}.mp4")

        # Extract video stream only
        subprocess.run([
            "ffmpeg", "-y",
            "-i", video_file,
            "-an",
            "-vcodec", "copy",
            v_path,
            "-hide_banner", "-loglevel", "error"
        ], check=True, capture_output=True, text=True)
        # Extract audio stream only
        subprocess.run([
            "ffmpeg", "-y",
            "-i", video_file,
            "-vn",
            "-acodec", "copy",
            a_path,
            "-hide_banner", "-loglevel", "error"
        ], check=True, capture_output=True, text=True)
        # Create low-resolution version
        subprocess.run([
            "ffmpeg", "-y",
            "-i", video_file,
            "-vf", f"scale='if(gt(iw,ih),-2,{args.target_max_dim})':'if(gt(iw,ih),{args.target_max_dim},-2)',fps={args.target_fps}",
            "-c:v", "libx264",
            "-c:a", "copy",
            "-preset", "fast",
            low_path,
            "-hide_banner", "-loglevel", "error"
        ], check=True, capture_output=True, text=True)

        item["v_path"] = os.path.relpath(v_path, args.root_path)
        item["a_path"] = os.path.relpath(a_path, args.root_path)
        item["low_path"] = os.path.relpath(low_path, args.root_path)
        return item
    except subprocess.CalledProcessError as e:
        print(f"error processing {id}: {e.stderr}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred with video {id}: {e}")
        return None


if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.script_folder, exist_ok=True)
    os.makedirs(args.sep_save_folder, exist_ok=True)
    os.makedirs(args.low_save_folder, exist_ok=True)

    with open(args.videos_list_file, "r", encoding="utf-8") as f:
        data_tmp = [json.loads(line) for line in f.readlines()]

    if os.path.exists(args.sep_videos_list_file):
        with open(args.sep_videos_list_file, "r", encoding="utf-8") as f:
            data_tmp += [json.loads(line) for line in f.readlines()]

    videos_list = dict()
    for item in data_tmp:
        videos_list[item["id"]] = item

    tasks = []
    for item in videos_list.values():
        if "v_path" in item and os.path.exists(os.path.join(args.root_path, item["v_path"])) and \
            "a_path" in item and os.path.exists(os.path.join(args.root_path, item["a_path"])) and \
                "low_path" in item and os.path.exists(os.path.join(args.root_path, item["low_path"])):
            continue
        tasks.append(item)

    worker_func = partial(process_video, args = args)
    with Pool(processes=args.num_processes) as pool:
        with open(args.sep_videos_list_file, "a", encoding="utf-8") as f:
            for result in tqdm(pool.imap_unordered(worker_func, tasks), total=len(tasks)):
                if result:
                    f.write(json.dumps(result) + "\n")
                    f.flush()
