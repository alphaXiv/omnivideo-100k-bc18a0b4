"""Score the data engine's output and write EVAL.md + artifacts.

Reproduces the paper's core, automatically-measurable claims for the data engine:

1. Entity-Anchored Video Scripting produced a structured, multi-segment script
   per video (summary + main entity list + per-segment AUDIO/VISUAL + speaker
   labels). We report counts and dump the scripts.

2. Clue-Guided QA Generation yields longer-temporal-span, more cross-segment QA
   than direct single-pass generation (paper Section 5.4: 144.75s vs 76.24s avg
   span). We measure the temporal span of the segment timestamps each method
   cites, on the SAME scripts, and compare.

3. Cross-segment ANSWERABILITY: a stronger test of the paper's underlying claim
   that clue-guided QA *requires* synthesizing multiple segments. The model can
   fabricate cited timestamps (mechanism 2 only measures what it *says* it
   used), so we additionally re-prompt the same model with the QA's question
   given (a) ONE random segment description vs (b) the FULL script, judge each
   answer against the generator's gold answer (Gemini-as-judge), and report the
   accuracy GAP per method. Clue-guided QA should show a much larger
   full-vs-single gap than direct QA."""

import os
import re
import json
import glob
import random
import hashlib

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


# ---------------------------------------------------------------------------
# Cross-segment answerability metric
# ---------------------------------------------------------------------------

QA_RE = re.compile(
    r"Q\s*:\s*(.+?)\s*A\s*:\s*(.+?)(?=\n[A-Z][A-Z_ ]*\s*:|\Z)",
    re.DOTALL,
)


def _mmss_to_seconds(mmss):
    mm, ss = mmss.split(":")
    return int(mm) * 60 + int(ss)


def format_segment(seg):
    """Render one segment in the same shape generate_qa.py feeds the model."""
    s = "[%s - %s]\n" % (seg.get("start_time", ""), seg.get("end_time", ""))
    audio = sorted(
        seg.get("transcription", []) + seg.get("non_speech", []),
        key=lambda x: (_mmss_to_seconds(x["start_time"]), _mmss_to_seconds(x["end_time"])),
    )
    astr = ""
    for a in audio:
        if "text" in a:
            astr += "(%s-%s) [%s]: %s\n" % (a["start_time"], a["end_time"],
                                            a.get("speaker", ""), a["text"])
        if "sound" in a:
            astr += "(%s-%s) (%s)\n" % (a["start_time"], a["end_time"], a["sound"])
    if astr:
        s += "AUDIO:\n" + astr
    vstr = ""
    for v in seg.get("visual", []):
        if "text" in v:
            vstr += "(%s-%s)\n%s\n" % (v["start_time"], v["end_time"],
                                         v["text"].replace("\n\n", "\n"))
    if vstr:
        s += "VISUAL:\n" + vstr
    return s


def format_all_segments(item):
    return "".join(format_segment(s) for s in item.get("segments", []))


def extract_qa(text):
    if not text:
        return None
    cleaned = text.replace("*", "").replace("#", "")
    m = QA_RE.search(cleaned)
    if not m:
        return None
    q = m.group(1).strip()
    a = m.group(2).strip()
    # Trim trailing USED_SEGMENTS / Analysis blocks that slipped past lookahead.
    a = re.split(r"\n(?:USED_SEGMENTS|Analysis|Correct Sequence)\s*:", a, maxsplit=1)[0].strip()
    if not q or not a:
        return None
    return q, a


def collect_qas(scripts_by_id):
    """Returns list of {method, task, video_id, group_idx, question, answer}."""
    out = []
    # Clue-guided: only the cross-segment *_qa.jsonl files have qa as a LIST of
    # groups whose `content` is the Q/A pair. Single-segment task files store
    # qa as a string and are not part of this comparison.
    for path in sorted(glob.glob(os.path.join(ROOT, "qa_files", "*_qa.jsonl"))):
        task = os.path.basename(path).replace("_qa.jsonl", "")
        if task.startswith("direct_"):
            continue
        latest = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                latest[item["id"]] = item  # later writes supersede earlier ones
        for vid, item in latest.items():
            qa = item.get("qa")
            if not isinstance(qa, list):
                continue  # single-segment tasks: not part of the comparison
            if vid not in scripts_by_id:
                continue
            for gi, grp in enumerate(qa):
                parsed = extract_qa(grp.get("content"))
                if not parsed:
                    continue
                out.append({"method": "clue_guided", "task": task, "video_id": vid,
                            "group_idx": gi, "question": parsed[0], "answer": parsed[1]})

    # Direct baseline.
    for path in sorted(glob.glob(os.path.join(ROOT, "qa_files", "direct_*.jsonl"))):
        task = os.path.basename(path).replace("direct_", "").replace(".jsonl", "")
        with open(path, "r", encoding="utf-8") as f:
            for gi, line in enumerate(f):
                item = json.loads(line)
                if item["id"] not in scripts_by_id:
                    continue
                parsed = extract_qa(item.get("direct_text"))
                if not parsed:
                    continue
                out.append({"method": "direct", "task": task, "video_id": item["id"],
                            "group_idx": gi, "question": parsed[0], "answer": parsed[1]})
    return out


SINGLE_SEG_PROMPT = """
You are watching a video that has been transcribed. Only the following SINGLE
segment of the video's textual description is available to you. Answer the
question as well as you can using only this information. Give a single concise
sentence; do not say "I cannot tell" unless absolutely nothing in the segment
relates to the question.

# Segment
{SEGMENT}

# Question
{QUESTION}

# Answer
""".strip()


FULL_SCRIPT_PROMPT = """
You are watching a video that has been transcribed. The full structured
description (summary, main entities, per-segment audio + visual script) is
provided below. Answer the question in a single concise sentence.

# Video Summary
{VIDEO_SUMMARY}

# Main Entities
{MAIN_ENTITIES}

# Detailed Script
{SEGMENTS}

# Question
{QUESTION}

# Answer
""".strip()


JUDGE_PROMPT = """
You are a strict, fair QA judge. Decide whether the CANDIDATE answer is
factually equivalent to the GOLD answer for the given QUESTION. Allow paraphrase
and reasonable abbreviation. Treat the candidate as INCORRECT if it contradicts
the gold answer, omits the key fact the gold answer hinges on, or only restates
the question.

Output exactly one token: CORRECT or INCORRECT.

QUESTION: {QUESTION}
GOLD: {GOLD}
CANDIDATE: {CAND}

Verdict:
""".strip()


def _make_client():
    from google import genai  # repro/vendor shim
    return genai.Client(api_key=os.environ["API_KEY"])


def _call(client, prompt):
    try:
        resp = client.models.generate_content(
            model=os.environ["MODEL_NAME"], contents=prompt,
        )
        return (resp.text or "").strip()
    except Exception as e:  # noqa: BLE001
        print("[answerability] call failed:", repr(e)[:200])
        return ""


def _judge(client, question, gold, cand):
    if not cand:
        return False
    verdict = _call(client, JUDGE_PROMPT.format(QUESTION=question, GOLD=gold, CAND=cand))
    return verdict.strip().upper().startswith("CORRECT")


def _qa_key(qa):
    raw = "%s|%s|%s|%s|%s" % (qa["method"], qa["task"], qa["video_id"],
                              qa["group_idx"], qa["question"])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def cross_segment_answerability(scripts):
    """Reprompt the same model with (single segment) vs (full script) for every
    generated QA, judge against the gold answer, and report the accuracy gap."""
    if not os.environ.get("API_KEY") or not os.environ.get("MODEL_NAME"):
        print("[answerability] API_KEY/MODEL_NAME not set; skipping")
        return None

    scripts_by_id = {s["id"]: s for s in scripts}
    qas = collect_qas(scripts_by_id)
    if not qas:
        print("[answerability] no QAs found; skipping")
        return None

    max_per_method = int(os.environ.get("ANSWERABILITY_MAX_QA", "0") or 0)
    if max_per_method:
        capped = []
        counts = {}
        for qa in qas:
            c = counts.get(qa["method"], 0)
            if c >= max_per_method:
                continue
            counts[qa["method"]] = c + 1
            capped.append(qa)
        qas = capped

    # Resume from cache so re-runs are cheap.
    cache_path = os.path.join(ART, "answerability_cache.jsonl")
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    cache[rec["key"]] = rec
                except Exception:  # noqa: BLE001
                    pass

    client = _make_client()
    rows = []
    with open(cache_path, "a", encoding="utf-8") as cache_f:
        for qa in qas:
            key = _qa_key(qa)
            rec = cache.get(key)
            if rec is None:
                item = scripts_by_id[qa["video_id"]]
                segs = item.get("segments", [])
                if not segs:
                    continue
                rng = random.Random(key)
                seg = rng.choice(segs)
                seg_idx = segs.index(seg)
                seg_text = format_segment(seg)
                full_text = format_all_segments(item)
                ents = "".join("- %s: %s\n" % (e["entity"], e["description"])
                               for e in item.get("main_entities", []))
                summary = (item.get("video_summary") or "").strip()

                single_ans = _call(client, SINGLE_SEG_PROMPT.format(
                    SEGMENT=seg_text.strip(), QUESTION=qa["question"]))
                full_ans = _call(client, FULL_SCRIPT_PROMPT.format(
                    VIDEO_SUMMARY=summary, MAIN_ENTITIES=ents.strip(),
                    SEGMENTS=full_text.strip(), QUESTION=qa["question"]))
                single_ok = _judge(client, qa["question"], qa["answer"], single_ans)
                full_ok = _judge(client, qa["question"], qa["answer"], full_ans)

                rec = {
                    "key": key, "method": qa["method"], "task": qa["task"],
                    "video_id": qa["video_id"], "group_idx": qa["group_idx"],
                    "question": qa["question"], "gold": qa["answer"],
                    "chosen_segment_idx": seg_idx,
                    "chosen_segment_time": "%s-%s" % (seg.get("start_time", ""),
                                                       seg.get("end_time", "")),
                    "single_answer": single_ans, "single_correct": bool(single_ok),
                    "full_answer": full_ans, "full_correct": bool(full_ok),
                }
                cache_f.write(json.dumps(rec) + "\n")
                cache_f.flush()
                cache[key] = rec
                print("[answerability] %s/%s %s g%d single=%s full=%s" % (
                    rec["method"], rec["task"], rec["video_id"], rec["group_idx"],
                    rec["single_correct"], rec["full_correct"]))
            rows.append(rec)

    def stats(method):
        m = [r for r in rows if r["method"] == method]
        if not m:
            return {"n": 0, "single_acc": None, "full_acc": None, "gap": None}
        n = len(m)
        s = sum(1 for r in m if r["single_correct"]) / n
        fa = sum(1 for r in m if r["full_correct"]) / n
        return {"n": n, "single_acc": round(s, 3),
                "full_acc": round(fa, 3), "gap": round(fa - s, 3)}

    cg = stats("clue_guided")
    dr = stats("direct")
    gap_ratio = None
    if cg.get("gap") is not None and dr.get("gap") not in (None, 0):
        gap_ratio = round(cg["gap"] / dr["gap"], 2)
    return {"clue_guided": cg, "direct": dr,
            "gap_ratio_clue_over_direct": gap_ratio,
            "n_evaluated": len(rows)}


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

    # --- mechanism 3: cross-segment answerability (single seg vs full script) ---
    answerability = cross_segment_answerability(scripts)

    result = {
        "model": os.environ.get("MODEL_NAME"),
        "n_videos": len(scripts),
        "scripts": script_summary,
        "clue_guided": {"n_qa": len(cg), "avg_span_s": round(cg_span, 1), "avg_timestamps": round(cg_ts, 2)},
        "direct": {"n_qa": len(dr), "avg_span_s": round(dr_span, 1), "avg_timestamps": round(dr_ts, 2)},
        "span_ratio_clue_over_direct": round(cg_span / dr_span, 2) if dr_span else None,
        "answerability": answerability,
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
    # per-QA answerability judgements (deduplicated snapshot of the cache)
    cache_path = os.path.join(ART, "answerability_cache.jsonl")
    if os.path.exists(cache_path):
        seen = {}
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                seen[rec.get("key")] = rec
        with open(os.path.join(ART, "answerability.jsonl"), "w", encoding="utf-8") as f:
            for rec in seen.values():
                f.write(json.dumps(rec) + "\n")

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

    # --- mechanism 3: cross-segment answerability ----------------------------
    lines.append("## Mechanism 3: Cross-Segment Answerability\n")
    lines.append("Cited timestamp spans (mechanism 2) only measure what the generator "
                 "*claims* it used; the model can fabricate citations. This metric "
                 "directly tests the underlying claim by re-asking the same model "
                 "each generated question given (a) ONE randomly chosen segment "
                 "description vs (b) the FULL script, and judging each answer against "
                 "the generator's gold answer (Gemini-as-judge). A clue-guided QA "
                 "that truly requires cross-segment synthesis should be answerable "
                 "from the full script but NOT from any single segment, so the "
                 "full-vs-single accuracy gap should be much larger for clue-guided "
                 "than for direct QA.\n")
    if answerability is None:
        lines.append("_Skipped (no QAs available, or API not configured)._\n")
    else:
        lines.append("| method | #QA evaluated | single-segment acc | full-script acc | gap (full - single) |")
        lines.append("|---|---|---|---|---|")
        for key, label in [("clue_guided", "clue-guided"), ("direct", "direct")]:
            s = answerability[key]
            if s.get("n", 0) == 0:
                lines.append(f"| {label} | 0 | - | - | - |")
            else:
                lines.append(f"| {label} | {s['n']} | {s['single_acc']} | {s['full_acc']} | {s['gap']} |")
        gr = answerability.get("gap_ratio_clue_over_direct")
        if gr is not None:
            lines.append("")
            lines.append(f"**Clue-guided / direct gap ratio: {gr}** "
                         "(>1 means clue-guided QA depends on cross-segment synthesis "
                         "more than direct QA does).\n")
        else:
            lines.append("")
            lines.append("_Gap ratio undefined (direct gap is zero or missing)._\n")

    eval_md = "\n".join(lines) + "\n"
    with open(os.path.join(ART, "EVAL.md"), "w", encoding="utf-8") as f:
        f.write(eval_md)
    with open(os.path.join(os.environ.get("REPO_ROOT", "."), "EVAL.md"), "w", encoding="utf-8") as f:
        f.write(eval_md)

    print(eval_md)
    print("[metrics] wrote artifacts to", ART)


if __name__ == "__main__":
    main()
