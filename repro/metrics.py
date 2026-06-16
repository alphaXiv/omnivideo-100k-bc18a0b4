"""Score the data engine's output and write EVAL.md + artifacts.

Reproduces the paper's core, automatically-measurable claims for the data engine:

1. Entity-Anchored Video Scripting produced a structured, multi-segment script
   per video (summary + main entity list + per-segment AUDIO/VISUAL + speaker
   labels). We report counts and dump the scripts.

2. Clue-Guided QA Generation yields longer-temporal-span, more cross-segment QA
   than direct single-pass generation (paper Section 5.4: 144.75s vs 76.24s avg
   span). We measure the temporal span of the segment timestamps each method
   cites, on the SAME scripts, and compare."""

import os
import re
import json
import glob

ROOT = os.environ["ROOT_PATH"]
ART = os.path.join(os.environ.get("REPO_ROOT", "."), ".openresearch", "artifacts")
os.makedirs(ART, exist_ok=True)

TS = re.compile(r"(\d{1,2}):(\d{2})")


def spans_from_text(text):
    """All MM:SS timestamps in a blob -> (min, max, count_distinct)."""
    secs = sorted({int(m) * 60 + int(s) for m, s in TS.findall(text or "")})
    if len(secs) < 2:
        return (secs[0] if secs else None, secs[0] if secs else None, len(secs))
    return secs[0], secs[-1], len(secs)


def load_scripts():
    items = []
    p = os.path.join(ROOT, "script.jsonl")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                items.append(json.loads(line))
    return items


def clue_guided_spans():
    """For each cross-segment task's *_segment_groups.jsonl, each clue group's
    'designated_segments' lists the segment timestamps it links. Span = max-min."""
    rows = []
    for path in sorted(glob.glob(os.path.join(ROOT, "qa_files", "*_segment_groups.jsonl"))):
        task = os.path.basename(path).replace("_segment_groups.jsonl", "")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                qa = item.get("qa")
                if not qa:
                    continue
                for grp in qa:
                    lo, hi, n = spans_from_text(grp.get("designated_segments", ""))
                    if lo is None:
                        continue
                    rows.append({"task": task, "id": item["id"], "span": hi - lo, "n_ts": n})
    return rows


def direct_spans():
    rows = []
    for path in sorted(glob.glob(os.path.join(ROOT, "qa_files", "direct_*.jsonl"))):
        task = os.path.basename(path).replace("direct_", "").replace(".jsonl", "")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                used = ""
                m = re.search(r"USED_SEGMENTS:(.*)", item.get("direct_text", ""), re.DOTALL)
                if m:
                    used = m.group(1)
                lo, hi, n = spans_from_text(used)
                if lo is None:
                    continue
                rows.append({"task": task, "id": item["id"], "span": hi - lo, "n_ts": n})
    return rows


def avg(xs):
    return sum(xs) / len(xs) if xs else 0.0


def main():
    scripts = load_scripts()

    # --- mechanism 1: scripting structure ---
    script_summary = []
    for it in scripts:
        n_seg = len(it.get("segments", []))
        n_ent = len(it.get("main_entities", []))
        n_spk = len({t.get("speaker") for s in it.get("segments", [])
                     for t in s.get("transcription", []) if t.get("speaker")})
        n_sounds = sum(len(s.get("non_speech", [])) for s in it.get("segments", []))
        script_summary.append({"id": it["id"], "duration_s": it.get("duration"),
                               "segments": n_seg, "main_entities": n_ent,
                               "speakers": n_spk, "non_speech_events": n_sounds})

    # --- mechanism 2: clue-guided vs direct temporal span ---
    cg = clue_guided_spans()
    dr = direct_spans()
    cg_span, dr_span = avg([r["span"] for r in cg]), avg([r["span"] for r in dr])
    cg_ts, dr_ts = avg([r["n_ts"] for r in cg]), avg([r["n_ts"] for r in dr])

    result = {
        "model": os.environ.get("MODEL_NAME"),
        "n_videos": len(scripts),
        "scripts": script_summary,
        "clue_guided": {"n_qa": len(cg), "avg_span_s": round(cg_span, 1), "avg_timestamps": round(cg_ts, 2)},
        "direct": {"n_qa": len(dr), "avg_span_s": round(dr_span, 1), "avg_timestamps": round(dr_ts, 2)},
        "span_ratio_clue_over_direct": round(cg_span / dr_span, 2) if dr_span else None,
    }
    with open(os.path.join(ART, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    # full scripts + per-QA spans as inspectable artifacts
    with open(os.path.join(ART, "scripts.jsonl"), "w", encoding="utf-8") as f:
        for it in scripts:
            f.write(json.dumps(it) + "\n")
    with open(os.path.join(ART, "qa_spans.jsonl"), "w", encoding="utf-8") as f:
        for r in cg:
            f.write(json.dumps({**r, "method": "clue_guided"}) + "\n")
        for r in dr:
            f.write(json.dumps({**r, "method": "direct"}) + "\n")

    # EVAL.md (the conventional artifact) + repo-root copy
    lines = []
    lines.append("# OmniVideo-100K data-engine reproduction\n")
    lines.append(f"Model (via OpenRouter): `{result['model']}`\n")
    lines.append("## Mechanism 1: Entity-Anchored Video Scripting\n")
    lines.append("| video | duration | segments | main entities | speakers | non-speech |")
    lines.append("|---|---|---|---|---|---|")
    for s in script_summary:
        lines.append(f"| {s['id']} | {s['duration_s']}s | {s['segments']} | "
                     f"{s['main_entities']} | {s['speakers']} | {s['non_speech_events']} |")
    lines.append("")
    lines.append("## Mechanism 2: Clue-Guided vs Direct QA generation\n")
    lines.append("Average temporal span of segments each method links per QA "
                 "(paper Section 5.4: 144.75s clue-guided vs 76.24s direct).\n")
    lines.append("| method | #QA | avg temporal span (s) | avg #timestamps linked |")
    lines.append("|---|---|---|---|")
    lines.append(f"| clue-guided | {result['clue_guided']['n_qa']} | "
                 f"{result['clue_guided']['avg_span_s']} | {result['clue_guided']['avg_timestamps']} |")
    lines.append(f"| direct | {result['direct']['n_qa']} | "
                 f"{result['direct']['avg_span_s']} | {result['direct']['avg_timestamps']} |")
    lines.append("")
    lines.append(f"**Clue-guided / direct span ratio: {result['span_ratio_clue_over_direct']}** "
                 "(paper ratio 144.75/76.24 = 1.90).\n")
    eval_md = "\n".join(lines) + "\n"
    with open(os.path.join(ART, "EVAL.md"), "w", encoding="utf-8") as f:
        f.write(eval_md)
    with open(os.path.join(os.environ.get("REPO_ROOT", "."), "EVAL.md"), "w", encoding="utf-8") as f:
        f.write(eval_md)

    print(eval_md)
    print("[metrics] wrote artifacts to", ART)


if __name__ == "__main__":
    main()
