import os
import re
import json
import string
import argparse
from tqdm import tqdm


# avoid pattern error, if check ok, set False, run again
check_format = True

cross_segment_oe = ["causal_reasoning", "future_prediction", "summarization",
                    "sentiment_analysis", "hypothetical_reasoning", "comparison"]
cross_segment_ordering = ["event_sequence_ordering"]
single_segment_oe = ["scene_transformation_detection"]
single_segment_subtask = ["fine_grained_perception", "context_understanding"]


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_path", type=str, required=True)

    args = parser.parse_args()
    args.script_file = os.path.join(args.root_path, "script.jsonl")
    args.qa_folder = os.path.join(args.root_path, "qa_files")
    return args


def get_data(qa_file):
    data_dict = {}
    if os.path.exists(qa_file):
        with open(qa_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                item = json.loads(line)
                data_dict[item["id"]] = item

    data = []
    for item in data_dict.values():
        if item["qa"] and item["qa"] != "NONE":
            data.append(item)
    return data


def get_save_json(question_id, item, task, Q, A, analysis, subtask=None, Options=None):
    data = {"id": item["id"],
            "question_id": question_id,
            "video_path": item["video_path"],
            "duration": item["duration"],
            "search_tag": item.get("search_tag", None),
            "language": item.get("language", None),
            "metadata": item.get("metadata", None),
            "resolution": item.get("resolution", None),
            "task": task,
            "subtask": subtask,
            "Q": Q,
            "Options": None,
            "A": A,
            "analysis": analysis}
    if Options:
        data["Options"] = Options
    else:
        data.pop("Options")
    if subtask:
        data["subtask"] = subtask
    else:
        data.pop("subtask")

    return json.dumps(data) + "\n"


def check_question(Q):
    if Q[0] in ["*", "#"] or Q[-1] not in ["?", ".", "\"", ":", ")"]:
        print("[ERROR] Check failed")
        print(Q)
        return False
    return True


def check_options(Options):
    if len(Options) <= 1:
        return False
    for option in Options:
        if option[0] in ["*", "#"] or option[-1] in ["*", "#"]:
            print("[ERROR] Check failed")
            print(option)
            return False
    return True


def check_answer_oe(A):
    if A[0] in ["*", "#"] or A[-1] not in [".", "\"", "'", "!", "?"]:
        print("[ERROR] Check failed")
        print(A)
        return False
    return True


def check_answer_ordering(Options, A):
    if len(A) != len(Options):
        print("[ERROR] Check failed")
        print(A)
        return False
    char_options = list(string.ascii_uppercase)[: len(A)]
    cond1 = all(c in char_options for c in A)
    cond2 = len(A) == len(set(A))
    if cond1 and cond2:
        return True
    else:
        print("[ERROR] Check failed")
        print(A)
        return False


if __name__ == "__main__":
    args = get_args()

    for task in cross_segment_oe:
        qa_file = os.path.join(args.qa_folder, f"{task}_qa.jsonl")
        data = get_data(qa_file)
        if not data:
            continue

        print(f"{task} begin")
        save_file = os.path.join(args.root_path, f"{task}.jsonl")
        with open(save_file, "w", encoding="utf-8") as f_out:
            for item in tqdm(data):
                qa_idx = 0
                for i in item["qa"]:
                    if "content" not in i or not i["content"]:
                        continue
                       
                    i["content"] = i["content"].replace("*", "").replace("#", "")
                    pattern = r"Q:\s*(.*?)\s*A:\s*(.*)"
                    match = re.search(pattern, i["content"], re.DOTALL)
                    if match:
                        Q = match.group(1).strip()
                        A = match.group(2).strip()
                        if not check_format or (check_question(Q) and check_answer_oe(A)):
                            i.pop("content")
                            text = get_save_json(f"{item["id"]}_{task}_{qa_idx}", item, task, Q, A, i)
                            f_out.write(text)
                            f_out.flush()
                            qa_idx += 1
                        else:
                            print(i["content"])
                    else:
                        print("[ERROR] Match failed")
                        print(i["content"])
        print(f"{task} completed")

    for task in cross_segment_ordering:
        qa_file = os.path.join(args.qa_folder, f"{task}_qa.jsonl")
        data = get_data(qa_file)
        if not data:
            continue

        print(f"{task} begin")
        save_file = os.path.join(args.root_path, f"{task}.jsonl")
        with open(save_file, "w", encoding="utf-8") as f_out:
            for item in tqdm(data):
                qa_idx = 0
                for i in item["qa"]:
                    if "content" not in i or not i["content"]:
                        continue

                    i["content"] = i["content"].replace("*", "").replace("#", "")
                    pattern = r"Q:\s*(.*?)\s*Events:\s*(.*?)\s*Correct Sequence:\s*(.*)"
                    match = re.search(pattern, i["content"], re.DOTALL)
                    if match:
                        Q = match.group(1).strip()
                        options_block = match.group(2).strip()
                        answer_block = match.group(3).strip()
                                
                        event_pattern = r"[A-Z]\.\s*(.*?)(?=\n[A-Z]\.|$)"
                        Options = re.findall(event_pattern, options_block, re.DOTALL)
                        if not Options:
                            print("[ERROR] Options Match failed")
                            print(i["content"])
                            continue
                            
                        A = re.findall(r"[A-Z]", answer_block)
                        if not A:
                            print("[ERROR] A Match failed")
                            print(i["content"])
                            continue
                        
                        if not check_format or (check_question(Q) and \
                            check_answer_ordering(Options, A) and check_options(Options)):
                            i.pop("content")
                            text = get_save_json(f"{item["id"]}_{task}_{qa_idx}", item, task, Q, A, i, Options=Options)
                            f_out.write(text)
                            f_out.flush()
                            qa_idx += 1
                        else:
                            print(i["content"])
                    else:
                        print("[ERROR] Match failed")
                        print(i["content"])
        print(f"{task} completed")

    for task in single_segment_oe:
        qa_file = os.path.join(args.qa_folder, f"{task}_qa.jsonl")
        data = get_data(qa_file)
        if not data:
            continue

        print(f"{task} begin")
        save_file = os.path.join(args.root_path, f"{task}.jsonl")
        with open(save_file, "w", encoding="utf-8") as f_out:
            for item in tqdm(data):
                item["qa"] = item["qa"].replace("*", "").replace("#", "")
                pattern = r"Q[1-2]:\s*(.*?)\s*A[1-2]:\s*(.*?)\s*Analysis:\s*(.*?)(?=\nQ[1-2]:|$)"
                matches = re.findall(pattern, item["qa"], re.DOTALL)
                if matches:
                    qa_idx = 0
                    for Q, A, analysis in matches:
                        if Q == "NONE":
                            continue

                        if not check_format or (check_question(Q) and check_answer_oe(A)):
                            text = get_save_json(f"{item["id"]}_{task}_{qa_idx}", item, task, Q, A, analysis)
                            f_out.write(text)
                            f_out.flush()
                            qa_idx += 1
                        else:
                            print(item["qa"])
                else:
                    print("[ERROR] Match failed")
                    print(item["qa"])
        print(f"{task} completed")

    for task in single_segment_subtask:
        qa_file = os.path.join(args.qa_folder, f"{task}_qa.jsonl")
        data = get_data(qa_file)
        if not data:
            continue

        print(f"{task} begin")
        save_file = os.path.join(args.root_path, f"{task}.jsonl")
        with open(save_file, "w", encoding="utf-8") as f_out:
            for item in tqdm(data):
                item["qa"] = item["qa"].replace("*", "").replace("#", "")
                if task == "fine_grained_perception":
                    block_pattern = r"(Audio-Guided|Vision-Guided)\s+(.*?)(?=\n(?:Audio-Guided|Vision-Guided)|$)"
                elif task == "context_understanding":
                    block_pattern = r"(Visual Context|Audio Context)\s+(.*?)(?=\n(?:Visual Context|Audio Context)|$)"

                matches = re.findall(block_pattern, item["qa"], re.DOTALL)
                if matches:
                    qa_idx = 0
                    for type, content in matches:
                        if "NONE" in content:
                            continue

                        pattern = r"Q:\s*(.*?)\s*A:\s*(.*?)\s*Analysis:\s*(.*)"
                        match = re.search(pattern, content, re.DOTALL)
                        if match:
                            Q = match.group(1).strip()
                            A = match.group(2).strip()
                            analysis = match.group(3).strip()

                            if not check_format or (check_question(Q) and check_answer_oe(A)):
                                text = get_save_json(f"{item["id"]}_{task}_{qa_idx}", item, task, Q, A, analysis, subtask=type)
                                f_out.write(text)
                                f_out.flush()
                                qa_idx += 1
                            else:
                                print(content)
                        else:
                            print("[ERROR] QA Match failed")
                            print(content)
                else:
                    print("[ERROR] Match failed")
                    print(item["qa"])
        print(f"{task} completed")
