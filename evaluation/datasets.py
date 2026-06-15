import os
import re
import json
import pandas as pd


MAX_DURATION = 300
DURATION_LIMIT_FOR_VIDEO_MME = ["short"]
DURATION_LIMIT_FOR_DAILY_OMNI = ["30s", "60s"]


def extract_model_answer(model_answer, num_options):
    possible_answers = [chr(65 + i) for i in range(num_options)]
    cleaned_text = ' '.join(model_answer.split())
    if not cleaned_text:
        return None
    found_prefixes = []

    for prefix in possible_answers:
        if re.search(rf'^{prefix}[.)\]:,]?(?:\s|$)', cleaned_text):
            found_prefixes.append(prefix.upper())

    regex_range = f"A-{possible_answers[-1]}"
    answer_patterns = [
        rf'(?:answer|option)(?: is)?\s*[:=-]?\s*["\'(]?\s*([{regex_range}])["\').]?',  # "answer: A", "answer is 'B'", "answer is (C)", "answer is "D""
        rf'Answer(?: is)?\s*[:=-]?\s*["\'(]?\s*([{regex_range}])\s*["\')]?',
        rf'([{regex_range}])(?:\s*is the|\)?\s*is)\s*(?:correct|right|answer)',  # "A is correct" or "B) is right" or "C is the answer"
        rf'select(?:ed)?\s*([{regex_range}])\b',  # "select B" or "selected D"
        rf'["\'(]\s*([{regex_range}])\s*["\')]\s*(?:is|as)\s*(?:the )?(?:answer|correct)',  # "'A' is the answer", "(B) is correct", '"C" as answer'
        rf"\bit's ([{regex_range}]).(?:\s|$)",  # "it's A."
        rf"I'd go with ([{regex_range}]).(?:\s|$)",  # "I'd go with A."
        rf'choose\s*:?([{regex_range}]).(?:\s|$)',  # "choose C."
        rf'option\s*([{regex_range}])\b',  # "option A" or "choose option C"
        rf'Option\s*([{regex_range}])\b',  # "Option A"
        rf'the correct order is:\s*([{regex_range}])\b',
        rf'It\'s ([{regex_range}]).',
    ]    
    for pattern in answer_patterns:
        matches = re.finditer(pattern, cleaned_text)
        for match in matches:
            extracted = match.group(1)
            if extracted in possible_answers:
                found_prefixes.append(extracted)

    if len(cleaned_text.split()) == 1:
        for prefix in possible_answers:
            if prefix == cleaned_text[0]:
                found_prefixes.append(prefix)
    
    if not found_prefixes:
        matches = re.finditer(rf"\bis ([{regex_range}]).(?:\s|$)", cleaned_text)
        for match in matches:
            extracted = match.group(1)
            if extracted in possible_answers:
                found_prefixes.append(extracted)
    
    if not found_prefixes:
        for prefix in possible_answers:
            if prefix == cleaned_text[-1]:
                found_prefixes.append(prefix)
            if len(cleaned_text) >= 2 and cleaned_text[-1] == "." and prefix == cleaned_text[-2]:
                found_prefixes.append(prefix)
    
    if not found_prefixes:
        for prefix in possible_answers:
            if re.search(rf'{prefix}.(?:\s|$)', cleaned_text):
                found_prefixes.append(prefix.upper())
    
    if not found_prefixes:
        for prefix in possible_answers:
            if re.search(rf'^{prefix}.', cleaned_text):
                found_prefixes.append(prefix.upper())

    if found_prefixes:
        return found_prefixes[0]
    return None


def check_correct(results_file, answer_key, options_key):
    valid_results = []
    with open(results_file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            item = json.loads(line)
            if options_key not in item:
                model_answer = extract_model_answer(item["model_answer"], 4)
            else:
                if isinstance(item[options_key], str):
                    item[options_key] = [opt.strip() for opt in item[options_key].split("\n") if opt.strip()]
                model_answer = extract_model_answer(item["model_answer"], len(item[options_key]))
            if model_answer is None:
                print(f"[Warning] Failed to extract answer from {item['model_answer']}")
            item["is_correct"] = model_answer == item[answer_key]
            valid_results.append(item)
    return valid_results


def print_overall(data):
    total_predictions = len(data)
    overall_correct = sum(1 for r in data if r.get("is_correct", False))
    overall_accuracy = overall_correct / total_predictions if total_predictions > 0 else 0
    print(f"Overall Accuracy: {overall_correct}/{total_predictions} = {overall_accuracy:.2%}")


def print_part(data, part_key):
    type_counts = dict()
    for result in data:
        type = result.get(part_key)
        type_counts.setdefault(type, {"total": 0, "correct": 0})
        type_counts[type]["total"] += 1
        if result.get("is_correct", False):
            type_counts[type]["correct"] += 1
    for type, counts in type_counts.items():
        type_total = counts["total"]
        type_correct = counts["correct"]
        type_accuracy = type_correct / type_total if type_total > 0 else 0
        print(f"    {type}: {type_correct}/{type_total} = {type_accuracy:.2%}")


class Video_MME:
    def __init__(self, dataset_dir, results_file):
        qa_file = os.path.join(dataset_dir, "videomme/test-00000-of-00001.parquet")
        videos_folder = os.path.join(dataset_dir, "video")
        self.results_file = results_file

        data_file = pd.read_parquet(qa_file)
        self.data = []
        for item in data_file.to_dict("records"):
            if item["duration"] in DURATION_LIMIT_FOR_VIDEO_MME:
                item["options"] = item["options"].tolist()
                self.data.append(item)

        processed_ids = set()
        if os.path.exists(results_file):
            with open(results_file, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    item = json.loads(line)
                    processed_ids.add(item["question_id"])

        self.unprocessed_data = []
        for item in self.data:
            if item["question_id"] not in processed_ids:
                video_path = os.path.join(videos_folder, f"{item['videoID']}.mp4")
                if not os.path.exists(video_path):
                    print(f"[Warning] Video file not found for ID {item['video_id']} at path {video_path}. Skipping.")
                    continue
                item["video_path"] = video_path
                
                FRAMES_TMPL_NOSUB = """
These are the frames of a video. \
Select the best answer to the following multiple-choice question based on the video. \
Respond with only the letter (A, B, C, or D) of the correct option.
"""
                item["question"] += "\n" + "\n".join(item["options"])
                prompt = FRAMES_TMPL_NOSUB + "Question: {}\nAnswer: ".format(item["question"])
                item["prompt"] = prompt
                self.unprocessed_data.append(item)
        print(f"[SUCCESS] Loaded Dataset Video-MME: {len(self.data)} questions (in {DURATION_LIMIT_FOR_VIDEO_MME}, total - {len(data_file)})")

    def calculate_accuracy(self):
        valid_results = check_correct(self.results_file, answer_key="answer", options_key="options")
        total_predictions = len(valid_results)
        if total_predictions == len(self.data):
            print("[SUCCESS] Evaluation completed.")
        else:
            print(f"[Warning] Evaluation uncompleted. Failed questions: {len(self.data) - total_predictions}")

        print_overall(valid_results)
        print("Accuracy by QA Type")
        print_part(valid_results, part_key="task_type")
        print("Accuracy by Video Duration")
        print_part(valid_results, part_key="duration")


class Daily_Omni:
    def __init__(self, dataset_dir, results_file):
        qa_file = os.path.join(dataset_dir, "qa.json")
        videos_folder = os.path.join(dataset_dir, "Videos")
        self.results_file = results_file

        with open(qa_file, "r", encoding="utf-8") as f:
            total_data = json.load(f)
        self.data = []
        for item in total_data:
            if item["video_duration"] in DURATION_LIMIT_FOR_DAILY_OMNI:
                self.data.append(item)

        processed_ids = set()
        if os.path.exists(results_file):
            with open(results_file, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    item = json.loads(line)
                    processed_ids.add(f"{item['video_id']}_{item['Question']}")

        self.unprocessed_data = []
        for item in self.data:
            item["question_id"] = f"{item['video_id']}_{item['Question']}"
            if item["question_id"] not in processed_ids:
                video_path = os.path.join(videos_folder, item["video_id"], f"{item['video_id']}_video.mp4")
                if not os.path.exists(video_path):
                    print(f"[Warning] Video file not found for ID {item['video_id']} at path {video_path}. Skipping.")
                    continue
                item["video_path"] = video_path

                prompt = f"""
Your task is to accurately answer multiple-choice questions based on the given video and audio together.
Select the single most accurate answer from the given choices.
Question: {item["Question"]}
Choices: {item["Choice"]}
Your answer should be a capital letter representing your choice: A, B, C, or D. Don't generate any other text.
"""
                item["prompt"] = prompt
                self.unprocessed_data.append(item)
        print(f"[SUCCESS] Loaded Dataset Daily-Omni: {len(self.data)} questions (in {DURATION_LIMIT_FOR_DAILY_OMNI}, total - {len(total_data)})")

    def calculate_accuracy(self):
        valid_results = check_correct(self.results_file, answer_key="Answer", options_key="Choice")
        total_predictions = len(valid_results)
        if total_predictions == len(self.data):
            print("[SUCCESS] Evaluation completed.")
        else:
            print(f"[Warning] Evaluation uncompleted. Failed questions: {len(self.data) - total_predictions}")

        print_overall(valid_results)
        print("Accuracy by QA Type")
        print_part(valid_results, part_key="Type")
        print("Accuracy by Video Duration")
        print_part(valid_results, part_key="video_duration")


class OmniVideoBench:
    def __init__(self, dataset_dir, results_file):
        qa_file = os.path.join(dataset_dir, "data.parquet")
        videos_folder = dataset_dir
        self.results_file = results_file
        
        # The following function is adapted from https://github.com/NJU-LINK/OmniVideoBench/blob/main/dataloader.py
        def convert_duration_to_seconds(time_str):
            """
            Converts a time string in 'MM:SS' or 'HH:MM:SS' format to total seconds.
                int: The total duration in seconds.
            """
            parts = time_str.split(':')
            seconds = 0
            if len(parts) == 2:  # MM:SS format
                minutes = int(parts[0])
                seconds = int(parts[1])
                total_seconds = minutes * 60 + seconds
            else:
                raise ValueError("Invalid time format. Please use 'MM:SS' or 'HH:MM:SS'.")
            
            return total_seconds

        data_file = pd.read_parquet(qa_file)
        data_file["reasoning_steps"] = data_file["reasoning_steps"].apply(list)
        data_file["options"] = data_file["options"].apply(list)
        self.data = []
        for item in data_file.to_dict("records"):
            item["duration"] = convert_duration_to_seconds(item["duration"])
            if item["duration"] > MAX_DURATION:
                continue
            self.data.append(item)

        processed_ids = set()
        if os.path.exists(results_file):
            with open(results_file, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    item = json.loads(line)
                    processed_ids.add(f"{item['video']}_{item['question']}")

        self.unprocessed_data = []
        for item in self.data:
            item["question_id"] = f"{item['video']}_{item['question']}"
            if item["question_id"] not in processed_ids:
                video_path = os.path.join(videos_folder, item["video"])
                if not os.path.exists(video_path):
                    print(f"[Warning] Video file not found for ID {item['video']} at path {video_path}. Skipping.")
                    continue
                item["video_path"] = video_path

                options_text = "\n".join(item["options"])
                prompt = (
                    "You are given a video. Based on the content of the video, answer the following question:\n\n"
                    f"Question:\n{item['question']}\n\n"
                    f"Options:\n{options_text}\n\n"
                    "Answer with the option's letter directly(e.g., A, B, C, or D)."
                    "If your access to the video content is limited, at least one option that is more likely than the others must be chosen."
                    "Mustn't give any other reason for can not choose!"
                )
                item["prompt"] = prompt
                self.unprocessed_data.append(item)
        print(f"[SUCCESS] Loaded Dataset OmniVideoBench: {len(self.data)} questions (<={MAX_DURATION}s, total - {len(data_file)})")

    def calculate_accuracy(self):
        valid_results = check_correct(self.results_file, answer_key="correct_option", options_key="options")

        questions = set()
        for item in self.data:
            questions.add(item["question_id"])
        new_results = []
        for valid_result in valid_results:
            if valid_result["question_id"] in questions:
                new_results.append(valid_result)
        valid_results = new_results

        total_predictions = len(valid_results)
        if total_predictions == len(self.data):
            print("[SUCCESS] Evaluation completed.")
        else:
            print(f"[Warning] Evaluation uncompleted. Failed questions: {len(self.data) - total_predictions}")

        print_overall(valid_results)
        print("Accuracy by QA Type")
        print_part(valid_results, part_key="question_type")
        print("Accuracy by Audio Type")
        print_part(valid_results, part_key="audio_type")
        print("Accuracy by Video Duration")
        for duration_threshold in [(0, 60), (60, 300), (300, 600), (600, 1800), (1800, MAX_DURATION)]:
            duration_filtered_results = [
                r for r in valid_results if r["duration"] > duration_threshold[0] and r["duration"] <= duration_threshold[1]
            ]
            duration_total = len(duration_filtered_results)
            duration_correct = sum(1 for r in duration_filtered_results if r.get("is_correct", False))
            duration_accuracy = duration_correct / duration_total if duration_total > 0 else 0
            print(f"    ({duration_threshold[0]} {duration_threshold[1]}]s: {duration_correct}/{duration_total} = {duration_accuracy:.2%}")


class JointAVBench:
    def __init__(self, dataset_dir, results_file):
        qa_file = os.path.join(dataset_dir, "jointavbench_fixed.json")
        videos_folder = os.path.join(dataset_dir, "videos")
        self.results_file = results_file
        
        # Due to the unknown errors of MiniCPM-o 4.5 in these samples,
        # in order to ensure the fairness of the comparison,
        # these data are not included in the evaluation.
        self.deprecated = ["TKGeXyYiLBo_task1_0", "z9ncrJM4gd4_task1_0", "STU9x7OxUy4_task2_4",
                           "T21p-RHFT3U_task2_1", "STU9x7OxUy4_task4_4", "TKGeXyYiLBo_task4_2",
                           "StCB7h6PjFw_task5_1", "T21p-RHFT3U_task5_0", "Rfos56t9HB4_task9_2"]

        with open(qa_file, "r", encoding="utf-8") as f:
            total_data = json.load(f)
        self.data = []
        for item in total_data:
            if item["qid"] in self.deprecated:
                continue

            start_time, end_time = item["segment_timestamp"]
            duration = end_time - start_time
            if duration > MAX_DURATION:
                continue
            
            item["duration"] = duration
            self.data.append(item)

        processed_ids = set()
        if os.path.exists(results_file):
            with open(results_file, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    item = json.loads(line)
                    processed_ids.add(item["qid"])

        self.unprocessed_data = []
        for item in self.data:
            item["question_id"] = item["qid"]
            if item["question_id"] not in processed_ids:
                video_path = os.path.join(videos_folder, f"{item['qid']}.mp4")
                if not os.path.exists(video_path):
                    print(f"[Warning] Video file not found for ID {item['qid']} at path {video_path}. Skipping.")
                    continue
                item["video_path"] = video_path

                prompt = "Watch the video carefully and answer the question with correct option letter (e.g., 'A').\nQuestion: " + item["prompt"]
                item["prompt"] = prompt
                self.unprocessed_data.append(item)
        print(f"[SUCCESS] Loaded Dataset JointAVBench: {len(self.data)} questions (<={MAX_DURATION}s, total - {len(total_data)})")

    def calculate_accuracy(self):
        valid_results = check_correct(self.results_file, answer_key="correct_prefix", options_key="options")
        
        new_results = []
        for valid_result in valid_results:
            if valid_result["question_id"] not in self.deprecated:
                new_results.append(valid_result)
        valid_results = new_results
            
        total_predictions = len(valid_results)
        if total_predictions == len(self.data):
            print("[SUCCESS] Evaluation completed.")
        else:
            print(f"[Warning] Evaluation uncompleted. Failed questions: {len(self.data) - total_predictions}")

        print_overall(valid_results)
        print("Accuracy by QA Type")
        print_part(valid_results, part_key="task")
        
        print("Accuracy by Scene Type")
        scene_task_dict = {
            "Single": ["STL", "VSSR", "SPL", "SOOG", "SOER", "SPER", "MPTI"],
            "Multiple": ["CSA", "MPO", "PDP", "AFA", "PTG"],
            "Full": ["AVDM", "MESI", "CRI"]
        }
        for item in valid_results:
            for scene_type, tasks in scene_task_dict.items():
                if item["task"] in tasks:
                    item["scene_type"] = scene_type
                    break
        print_part(valid_results, part_key="scene_type")
        
        cognitive_task_dict = {
            "Temporal": ["STL", "VSSR", "PTG"],
            "Spatial": ["SPL", "SOOG", "SOER"],
            "Emotion": ["SPER", "MPTI", "MESI"],
            "Long-form": ["CSA", "AVDM"],
            "Plot": ["MPO", "PDP", "AFA", "CRI"]
        }
        for item in valid_results:
            for cognitive_type, tasks in cognitive_task_dict.items():
                if item["task"] in tasks:
                    item["cognitive_type"] = cognitive_type
                    break
        print("Accuracy by Cognitive Type")
        print_part(valid_results, part_key="cognitive_type")


class FutureOmni:
    def __init__(self, dataset_dir, results_file):
        qa_file = os.path.join(dataset_dir, "futureomni_test.json")
        videos_folder = os.path.join(dataset_dir, "videos")
        self.results_file = results_file

        with open(qa_file, "r", encoding="utf-8") as f:
            total_data = json.load(f)
        self.data = []
        for item in total_data:
            if item["seconds"] > MAX_DURATION:
                continue
            
            item["duration"] = item["seconds"]
            self.data.append(item)

        processed_ids = set()
        if os.path.exists(results_file):
            with open(results_file, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    item = json.loads(line)
                    processed_ids.add(item["qid"])

        self.unprocessed_data = []
        for item in self.data:
            item["question_id"] = item["qid"]
            if item["question_id"] not in processed_ids:
                video_path = os.path.join(videos_folder, f"{item['qid']}.mp4")
                if not os.path.exists(video_path):
                    print(f"[Warning] Video file not found for ID {item['qid']} at path {video_path}. Skipping.")
                    continue
                item["video_path"] = video_path
                
                TEST_PROMPT_OMNI2 = """
These are the frames of a video and the corresponding audio.
Please answer the following multiple-choice question based on the video and audio content.
Choose the correct option and respond with ONLY the letter (A, B, C, D, E and F) of your choice.
"""
                question_str = item["question"] +  "\n".join(item["options"])
                prompt = TEST_PROMPT_OMNI2 + "Question: {}\nAnswer: ".format(question_str)
                item["prompt"] = prompt
                self.unprocessed_data.append(item)
        print(f"[SUCCESS] Loaded Dataset FutureOmni: {len(self.data)} questions (<={MAX_DURATION}s, total - {len(total_data)})")

    def calculate_accuracy(self):
        valid_results = check_correct(self.results_file, answer_key="answer", options_key="options")
        total_predictions = len(valid_results)
        if total_predictions == len(self.data):
            print("[SUCCESS] Evaluation completed.")
        else:
            print(f"[Warning] Evaluation uncompleted. Failed questions: {len(self.data) - total_predictions}")

        print_overall(valid_results)
        print("Accuracy by QA Type")
        print_part(valid_results, part_key="forecasting_pattern")
        print("Accuracy by Audio Type")
        print_part(valid_results, part_key="audio_type")
        print("Accuracy by Video Duration")
        for duration_threshold in [(0, 120), (120, 180), (180, 240), (240, MAX_DURATION)]:
            duration_filtered_results = [
                r for r in valid_results if r["duration"] > duration_threshold[0] and r["duration"] <= duration_threshold[1]
            ]
            duration_total = len(duration_filtered_results)
            duration_correct = sum(1 for r in duration_filtered_results if r.get("is_correct", False))
            duration_accuracy = duration_correct / duration_total if duration_total > 0 else 0
            print(f"  ({duration_threshold[0]} {duration_threshold[1]}]s: {duration_correct}/{duration_total} = {duration_accuracy:.2%}")


class OmniVideo_Test:
    def __init__(self, dataset_dir, results_file):
        qa_file = os.path.join(dataset_dir, "test_505.jsonl")
        videos_folder = os.path.join(dataset_dir, "videos")
        self.results_file = results_file

        with open(qa_file, "r", encoding="utf-8") as f:
            total_data = [json.loads(line) for line in f.readlines()]
        self.data = []
        for item in total_data:
            duration = item["end_time"] - item["start_time"]
            if duration > MAX_DURATION:
                continue
            
            item["duration"] = duration
            self.data.append(item)

        processed_ids = set()
        if os.path.exists(results_file):
            with open(results_file, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    item = json.loads(line)
                    processed_ids.add(item["question_id"])

        self.unprocessed_data = []
        for item in self.data:
            if item["question_id"] not in processed_ids:
                video_path = os.path.join(videos_folder, f"{item['question_id']}.mp4")
                if not os.path.exists(video_path):
                    print(f"[Warning] Video file not found for ID {item['question_id']} at path {video_path}. Skipping.")
                    continue
                item["video_path"] = video_path
                
                question = item["question"]
                options = item["options"]
                options_str = "\n".join([f"{chr(65+i)}. {option}" for i, option in enumerate(options)])
                prompt = f"""These are the frames of a video and the corresponding audio.
Select the best answer to the following multiple-choice question based on the video and audio content.
Respond with only the letter (A, B, C, or D) of the correct option.
Question: {question}\n{options_str}\nAnswer:
"""
                item["prompt"] = prompt
                self.unprocessed_data.append(item)
        print(f"[SUCCESS] Loaded Dataset: {len(self.data)} questions (<={MAX_DURATION}s, total - {len(total_data)})")

    def calculate_accuracy(self):
        valid_results = check_correct(self.results_file, answer_key="answer", options_key="options")
        total_predictions = len(valid_results)
        if total_predictions == len(self.data):
            print("[SUCCESS] Evaluation completed.")
        else:
            print(f"[Warning] Evaluation uncompleted. Failed questions: {len(self.data) - total_predictions}")

        print_overall(valid_results)
        print("Accuracy by QA Type")
        print_part(valid_results, part_key="task")
        print("Accuracy by Video Duration")
        for duration_threshold in [(0, 120), (120, MAX_DURATION)]:
            duration_filtered_results = [
                r for r in valid_results if (r["end_time"] - r["start_time"]) > duration_threshold[0] and (r["end_time"] - r["start_time"]) <= duration_threshold[1]
            ]
            duration_total = len(duration_filtered_results)
            duration_correct = sum(1 for r in duration_filtered_results if r.get("is_correct", False))
            duration_accuracy = duration_correct / duration_total if duration_total > 0 else 0
            print(f"  ({duration_threshold[0]} {duration_threshold[1]}]s: {duration_correct}/{duration_total} = {duration_accuracy:.2%}")
        print("Accuracy by Layer")
        for item in valid_results:
            if item["task"] in ["fine_grained_perception", "scene_transformation_detection"]:
                item["layer"] = "Perception"
            elif item["task"] in ["context_understanding", "comparison", "sentiment_analysis", "event_sequence_ordering", "summarization"]:
                item["layer"] = "Understanding"
            elif item["task"] in ["causal_reasoning", "future_prediction", "hypothetical_reasoning"]:
                item["layer"] = "Reasoning"
        print_part(valid_results, part_key="layer")


class Video_MME_v2:
    def __init__(self, dataset_dir, results_file):
        qa_file = os.path.join(dataset_dir, "test.parquet")
        videos_folder = os.path.join(dataset_dir, "video")
        self.results_file = results_file

        data_file = pd.read_parquet(qa_file)
        data_file["video"] = data_file["video_id"].apply(str)
        self.data = []

        import cv2
        def get_video_duration(video_path):
            if not os.path.exists(video_path):
                return None
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                rate = cap.get(cv2.CAP_PROP_FPS)
                frame_number = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                duration = frame_number / rate
                cap.release()
                return duration
            return None

        video_ids_duration = {}
        from tqdm import tqdm
        for item in tqdm(data_file.to_dict("records")):
            video_path = os.path.join(videos_folder, f"{item['video']}.mp4")
            if not os.path.exists(video_path):
                print(f"[Warning] Video file not found for ID {item['video']} at path {video_path}. Skipping.")
                continue
            item["video_path"] = video_path

            # self.data.append(item)
            if item["video"] in video_ids_duration:
                item["duration"] = video_ids_duration[item["video"]]
                self.data.append(item)
            else:
                duration = get_video_duration(video_path)
                if duration is None or duration > MAX_DURATION:
                    continue
                video_ids_duration[item["video"]] = duration
                item["duration"] = duration
                self.data.append(item)

        processed_ids = set()
        if os.path.exists(results_file):
            with open(results_file, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    item = json.loads(line)
                    processed_ids.add(item["question_id"])

        self.unprocessed_data = []
        for item in self.data:
            if item["question_id"] not in processed_ids:
                WO_SUB_PROMPT = "These are the frames of a video."
                INSTRUCT_PROMPT = (
                    "Select the best answer to the following multiple-choice question based on the video."
                    "Respond with only the letter (A, B, C, D, E, F, G, or H) of the correct option."
                )
                item["question"] += "\n" + item["options"]
                prompt = WO_SUB_PROMPT + f"Question: {item['question']}\n{INSTRUCT_PROMPT}"
                item["prompt"] = prompt
                self.unprocessed_data.append(item)
        print(f"[SUCCESS] Loaded Dataset Video-MME-v2: {len(self.data)} questions (<={MAX_DURATION}s, total - {len(data_file)})")

    def calculate_accuracy(self):
        results = check_correct(self.results_file, answer_key="answer", options_key="options")
        data = {}
        for item in self.data:
            data[item["question_id"]] = item
        valid_results = []
        for item in results:
            if item["question_id"] in data:
                valid_results.append(item)
        total_predictions = len(valid_results)
        if total_predictions == len(self.data):
            print("[SUCCESS] Evaluation completed.")
        else:
            print(f"[Warning] Evaluation uncompleted. Failed questions: {len(self.data) - total_predictions}")

        def get_final_rating(data):
            import ast

            all_groups = [[] for _ in range((len(data) + 1) // 4)]
            final_rating = {
                "level_1": [], "level_2": [], "level_3": [],
                "relevance_score": [], "relevance_linear_score": [],
                "logic_score": [], "total": [],
            }
            second_head_rating = {}
            third_head_rating = {}

            for i in range(len(data)):
                level = data[i]["level"]
                group_type = data[i]["group_type"]
                group_structure = data[i]["group_structure"]
                score = int(data[i]["is_correct"])
                second_head = data[i]["second_head"]
                third_head = data[i]["third_head"]
                all_groups[i // 4].append((level, group_type, group_structure, score, second_head, third_head))

            for group in all_groups:
                level = group[-1][0]
                group_type = group[-1][1]
                group_structure = group[-1][2]
                second_head = group[-1][4]
                third_head = group[-1][5]
                scores = [item[3] for item in group]
                
                def cal_relevance(scores):
                    score_map = {0: 0.0, 1: 100.0 / 16, 2: 100.0 * 4 / 16, 3: 100.0 * 9 / 16, 4: 100.0}
                    correct_count = sum(scores)
                    return score_map.get(correct_count, 0.0), correct_count * 25.0

                def cal_logic(scores, group_structure):
                    group_structure_list = ast.literal_eval(group_structure)
                    last_correct_idx = -1
                    for idx, val in enumerate(scores):
                        if val:
                            last_correct_idx = idx
                        else:
                            break
                    if group_structure_list == [1, 2, 3, 4]:
                        score_map = {0: 0.0, 1: 100.0 / 16, 2: 100.0 * 4 / 16, 3: 100.0 * 9 / 16, 4: 100.0}
                    elif group_structure_list == [1, [2, 3], 4]:
                        score_map = {0: 0.0, 1: 100.0 / 12, 2: 100.0 * 4 / 12, 3: 100.0 * 7 / 12, 4: 100.0}
                        if last_correct_idx == 0 and scores[2]:
                            last_correct_idx += 1
                    elif group_structure_list == [[1, 2], 3, 4]:
                        score_map = {0: 0.0, 1: 100.0 / 10, 2: 100.0 * 2 / 10, 3: 100.0 * 5 / 10, 4: 100.0}
                        if last_correct_idx == -1 and scores[1]:
                            last_correct_idx += 1
                    else:
                        raise ValueError(f"Unknown group_structure_list: {group_structure_list}")
                    return score_map.get(last_correct_idx + 1, 0.0)

                if group_type == "relevance":
                    exp_score, linear_score = cal_relevance(scores)
                    final_rating["relevance_score"].append(exp_score)
                    final_rating["relevance_linear_score"].append(linear_score)
                elif group_type == "logic":
                    exp_score = cal_logic(scores, group_structure)
                    final_rating["logic_score"].append(exp_score)
                else:
                    raise ValueError(f"Unknown group_type: {group_type}")

                if level is not None and str(level) != "None":
                    final_rating[f"level_{int(level)}"].append(exp_score)
                final_rating["total"].append(exp_score)

                if second_head not in second_head_rating:
                    second_head_rating[second_head] = []
                second_head_rating[second_head].append(exp_score)
                if third_head not in third_head_rating:
                    third_head_rating[third_head] = []
                third_head_rating[third_head].append(exp_score)

            for key in final_rating:
                vals = final_rating[key]
                final_rating[key] = sum(vals) / len(vals) if vals else 0.0
            for key in second_head_rating:
                vals = second_head_rating[key]
                second_head_rating[key] = sum(vals) / len(vals) if vals else 0.0
            for key in third_head_rating:
                vals = third_head_rating[key]
                third_head_rating[key] = sum(vals) / len(vals) if vals else 0.0

            print(f"\n{'Metric':<40} {'Score':>8}")
            print("-" * 40)
            for k, v in final_rating.items():
                print(f"{k:<40} {v:>8.2f}")

            if any(str(v) != "None" for v in second_head_rating.keys()):
                print(f"\n{'Second Head':<40} {'Score':>8}")
                print("-" * 40)
                for k, v in second_head_rating.items():
                    print(f"{str(k):<40} {v:>8.2f}")

            if any(str(v) != "None" for v in third_head_rating.keys()):
                print(f"\n{'Third Head':<40} {'Score':>8}")
                print("-" * 40)
                for k, v in third_head_rating.items():
                    print(f"{str(k):<40} {v:>8.2f}")
        get_final_rating(valid_results)
