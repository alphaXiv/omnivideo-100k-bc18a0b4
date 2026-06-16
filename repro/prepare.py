"""Fetch a couple of real OmniVideo-Test clips and write the
`pre_files/final_videos_list.jsonl` that the data engine (step 0) consumes.

These are genuine OmniVideo videos published with the paper
(huggingface.co/datasets/MiG-NJU/OmniVideo-Test). They are multi-minute, so the
Entity-Anchored scripting produces a multi-segment script and Clue-Guided QA
generation has cross-segment material to mine."""

import os
import json
import subprocess
import urllib.request

ROOT = os.environ["ROOT_PATH"]
HF = "https://huggingface.co/datasets/MiG-NJU/OmniVideo-Test/resolve/main/videos"

# (id, source filename on HF). ids avoid a leading dash for shell/ffmpeg safety.
# Five multi-minute clips covering all three cross-segment tasks the engine runs
# (causal_reasoning, comparison, summarization), each present at least once.
# More videos (and at least one longer clip per task) average out per-video
# variance in the clue-guided-vs-direct span comparison.
CLIPS = [
    ("sIlvsZag5fc_causal_reasoning_1", "sIlvsZag5fc_causal_reasoning_1.mp4"),
    ("7nYHHgLj4M8_causal_reasoning_0", "7nYHHgLj4M8_causal_reasoning_0.mp4"),
    ("8aKv3bgeVMs_comparison_0", "8aKv3bgeVMs_comparison_0.mp4"),
    ("LFnCcYAyB9s_comparison_0", "LFnCcYAyB9s_comparison_0.mp4"),
    ("0COmvK458s0_summarization_0", "0COmvK458s0_summarization_0.mp4"),
]


def ffprobe_duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", path],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return int(round(float(out)))


def download(url, dst):
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
            with urllib.request.urlopen(req, timeout=300) as r, open(dst, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            if os.path.getsize(dst) > 100_000:
                return
        except Exception as e:  # noqa: BLE001
            print("[retry download]", url, e)
    raise RuntimeError("failed to download %s" % url)


def main():
    ori = os.path.join(ROOT, "videos", "ori")
    pre = os.path.join(ROOT, "pre_files")
    os.makedirs(ori, exist_ok=True)
    os.makedirs(pre, exist_ok=True)

    records = []
    for vid, fname in CLIPS:
        dst = os.path.join(ori, "%s.mp4" % vid)
        if not (os.path.exists(dst) and os.path.getsize(dst) > 100_000):
            print("[download]", fname)
            download("%s/%s" % (HF, fname), dst)
        dur = ffprobe_duration(dst)
        rel = os.path.relpath(dst, ROOT)
        records.append({"id": vid, "video_path": rel, "duration": dur})
        print("[ready] %s  duration=%ss  path=%s" % (vid, dur, rel))

    with open(os.path.join(pre, "final_videos_list.jsonl"), "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print("[done] wrote final_videos_list.jsonl with %d videos" % len(records))


if __name__ == "__main__":
    main()
