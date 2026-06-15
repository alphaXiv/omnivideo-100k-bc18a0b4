import os
import torch


class Qwen25_Omni:
    def __init__(self, model_path, return_audio=False):
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
        from qwen_omni_utils import process_mm_info
        self.process_mm_info = process_mm_info

        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2"
        )

        if not return_audio:
            self.model.disable_talker()
        self.RETURN_AUDIO = return_audio

        print(f"[SUCCESS] {model_path} loaded.")
        print("[INFO] Default parameters:")
        print(f"  return_audio = {self.RETURN_AUDIO}")

    def inference(self, item, max_frames):
        messages = [{"role": "system",
                     "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]},
                    {"role": "user",
                     "content": [{"type": "video",
                                  "video": item["video_path"],
                                  "max_frames": max_frames},
                                 {"type": "text", "text": item["prompt"]}]}]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        audios, images, videos, video_kwargs = self.process_mm_info(messages, use_audio_in_video=True, return_video_kwargs=True)

        # Modify transformers/models/qwen2_5_omni/processing_qwen2_5_omni.py Line 185 and Line 187
        """
        fps = output_kwargs["videos_kwargs"].get("fps", [2.0] * len(video_grid_thw))
        second_per_grid_ts = [self.video_processor.temporal_patch_size / i for i in fps]
        """
        inputs = self.processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=True, fps=video_kwargs["fps"])
        inputs = inputs.to(self.model.device).to(self.model.dtype)
        input_len = inputs["input_ids"].shape[1]
        if input_len >= 32768:
            raise ValueError(f"Input length {input_len} exceeds model limit. Skipping.")
        
        # check input
        # print(f"[INFO] Duration: {item.get('duration', 'unknown')}")
        # print(f"[INFO] Ori Video: {videos[0].shape}")
        # print(f"[INFO] video_second_per_grid: {inputs['video_second_per_grid']}")
        # print(f"[INFO] Video: {inputs['video_grid_thw']}")
        # input_lengths = (inputs["feature_attention_mask"].sum(-1) - 1) // 2 + 1
        # audio_lengths = (input_lengths - 2) // 2 + 1
        # print(f"[INFO] Audio: {audio_lengths}")  # duration * 25
        # print(f"[INFO] Input token: {input_len}")  # video_grid_thw.prod() / 2**2 + audio_lengths

        output = self.model.generate(**inputs,
                                     use_audio_in_video=True,
                                     return_audio=self.RETURN_AUDIO,
                                     do_sample=False,
                                     num_beams=1)
        response = self.processor.batch_decode(output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return response


class Qwen3_Omni:
    def __init__(self, model_path, return_audio=False):
        from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
        from qwen_omni_utils import process_mm_info
        self.process_mm_info = process_mm_info

        self.processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2"
        )

        if not return_audio:
            self.model.disable_talker()
        self.RETURN_AUDIO = return_audio
    
        print(f"[SUCCESS] {model_path} loaded.")
        print("[INFO] Default parameters:")
        print(f"  return_audio = {self.RETURN_AUDIO}")

    def inference(self, item, max_frames):
        messages = [{"role": "user",
                     "content": [{"type": "video", "video": item["video_path"], "max_frames": max_frames},
                                 {"type": "text", "text": item["prompt"]}]}]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        audios, images, videos, video_kwargs = self.process_mm_info(messages, use_audio_in_video=True, return_video_kwargs=True)

        # modify Qwen3-Omni-30B-A3B-Instruct/preprocessor_config.json: Add "chunk_length": 300
        # Modify transformers/models/qwen3_omni_moe/processing_qwen3_omni_moe.py Line 199
        """
        if not isinstance(fps, list):
            fps = [fps] * len(videos)
        """
        inputs = self.processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=True, fps=video_kwargs["fps"])
        inputs = inputs.to(self.model.device).to(self.model.dtype)
        input_len = inputs["input_ids"].shape[1]
        if input_len >= 65536:
            raise ValueError(f"Input length {input_len} exceeds model limit. Skipping.")
        
        # check input
        # print(f"[INFO] Duration: {item.get('duration', 'unknown')}")
        # print(f"[INFO] Ori Video: {videos[0].shape}")
        # print(f"[INFO] video_second_per_grid: {inputs['video_second_per_grid']}")
        # print(f"[INFO] Video: {inputs['video_grid_thw']}")
        # input_lengths = inputs["feature_attention_mask"].sum(-1)
        # input_lengths_leave = input_lengths % 100
        # feat_lengths = (input_lengths_leave - 1) // 2 + 1
        # audio_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
        # print(f"[INFO] Audio: {audio_lengths}")  # duration * 13
        # print(f"[INFO] Input token: {input_len}")  # video_grid_thw.prod() / 2**2 + audio_lengths

        output, _ = self.model.generate(**inputs,
                                        thinker_return_dict_in_generate=True,
                                        use_audio_in_video=True,
                                        return_audio=self.RETURN_AUDIO,
                                        do_sample=False,
                                        num_beams=1)
        response = self.processor.batch_decode(output.sequences[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return response


class Uni_Moe_2_Omni:
    def __init__(self, model_path):
        from uni_moe.model import deepspeed_moe_inference_utils
        from uni_moe.model.processing_qwen2_vl import Qwen2VLProcessor
        from uni_moe.model.modeling_out import GrinQwen2VLOutForConditionalGeneration
        from uni_moe.qwen_vl_utils import process_mm_info
        self.process_mm_info = process_mm_info

        # Modify Uni-MoE-2/uni_moe/model/modeling_qwen_grin_moe.py Line 2332
        """
        target_device = all_aux_loss[0].device
        all_aux_loss = torch.mean(torch.cat([l.to(target_device).unsqueeze(0) for l in all_aux_loss]))
        """
        self.model = GrinQwen2VLOutForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2"
        )
        self.processor = Qwen2VLProcessor.from_pretrained(model_path)
        self.processor.data_args = self.model.config
        print(f"[SUCCESS] {model_path} loaded.")

    def inference(self, item, max_frames):
        # Modify Uni-MoE-2/uni_moe/model/processing_qwen2_vl.py Line 176: 32 -> MAX_FRAMES
        # Modify Uni-MoE-2/uni_moe/qwen_vl_utils/mm_process.py Line 323 for FutureOmni's 1s examples
        """
        try:
            sample = sample.to_soundarray(fps=sample_rate)
        except OSError as e:
            print(f"{e}\nAttempting recovery using librosa...")
            sample, sample_rate = librosa.load(ele["audio"], sr=None, mono=False)
            if sample.ndim > 1:
                sample = sample.T
            else:
                sample = sample[:, np.newaxis]
        """

        messages = [{"role": "user",
                     "content": [{"type": "audio", "audio": item["video_path"]},
                                 {"type": "video", "video": item["video_path"]},
                                 {"type": "text", "text": f"<video>\n<audio>\n{item['prompt']}"}]}]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        text = text.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>").replace("<audio>", "<|audio_start|><|audio_pad|><|audio_end|>").replace("<video>", "<|vision_start|><|video_pad|><|vision_end|>")
        images, videos, audios = self.process_mm_info(messages)

        inputs = self.processor(text=text, audios=audios, images=images, videos=videos, return_tensors="pt", padding=True)
        if "second_grid_ts" in inputs:
            inputs["second_per_grid_ts"] = inputs["second_grid_ts"]
            del inputs["second_grid_ts"]
        inputs["input_ids"] = inputs["input_ids"].unsqueeze(0)
        inputs = inputs.to(self.model.device)
        for k, v in inputs.items():
            if k in ["pixel_values", "pixel_values_videos", "audio_features"]:
                inputs[k] = v.to(dtype=torch.bfloat16)
        
        # check input
        # input_len = inputs["input_ids"].shape[1]
        # print(f"[INFO] Duration: {item.get('duration', 'unknown')}")
        # print(f"[INFO] Ori Video: {inputs['pixel_values_videos'].shape}")
        # print(f"[INFO] video_second_per_grid: {inputs['video_second_per_grid']}")
        # print(f"[INFO] Video: {inputs['video_grid_thw']}")
        # print(f"[INFO] Audio: {inputs['audio_grid_thw']}")  # duration / 30s
        # print(f"[INFO] Input token: {input_len}")  # video_grid_thw.prod() * 12**2 + audio_grid_thw * 200

        output = self.model.generate(**inputs,
                                     use_cache=True,
                                     pad_token_id=self.processor.tokenizer.eos_token_id,
                                     do_sample=False,
                                     num_beams=1)
        response = self.processor.batch_decode(output[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)[0]
        return response


class video_SALMONN2_plus:
    def __init__(self, model_path):
        # The following function is adapted from https://github.com/bytedance/video-SALMONN-2/blob/main/video_SALMONN2_plus/inference.py
        def apply_liger_kernel_to_qwen2_5_vl(
            rope: bool = True,
            cross_entropy: bool = False,
            fused_linear_cross_entropy: bool = True,
            rms_norm: bool = True,
            swiglu: bool = True,
        ) -> None:
            """
            Apply Liger kernels to replace original implementation in HuggingFace Qwen2.5-VL models.
            NOTE: Qwen2.5-VL is not available in transformers<4.48.2

            Args:
                cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
                fused_linear_cross_entropy (bool):
                    Whether to apply Liger's fused linear cross entropy loss. Default is True.
                    `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
                    If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
                rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
                swiglu (bool): Whether to apply Liger's SwiGLU MLP. Default is True.
                model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
                loaded. Default is None.
            """

            print("Applying Liger kernels to Qwen2.5-VL model...")

            assert not (cross_entropy and fused_linear_cross_entropy), (
                "cross_entropy and fused_linear_cross_entropy cannot both be True."
            )

            from qwenvl.model import modeling_qwen2_5_vl
            from liger_kernel.transformers.rms_norm import LigerRMSNorm
            from liger_kernel.transformers.swiglu import LigerSwiGLUMLP
            from liger_kernel.transformers.qwen2vl_mrope import liger_multimodal_rotary_pos_emb

            if rope:
                modeling_qwen2_5_vl.apply_multimodal_rotary_pos_emb = liger_multimodal_rotary_pos_emb
            if rms_norm:
                modeling_qwen2_5_vl.Qwen2RMSNorm = LigerRMSNorm
            if swiglu:
                modeling_qwen2_5_vl.Qwen2MLP = LigerSwiGLUMLP
        apply_liger_kernel_to_qwen2_5_vl()

        from transformers import AutoTokenizer
        from qwenvl.model.modeling_qwen2_5_vl import video_SALMONN2_plus

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            model_max_length=131072,
            padding_side="right",
            use_fast=False
        )
        self.model = video_SALMONN2_plus.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2"
        )

        def prepare_dataset(model_path):
            from qwenvl.train.argument import DataArguments
            from transformers import WhisperFeatureExtractor
            from qwenvl.data.image_processing_qwen2_vl_fast import Qwen2VLImageProcessorFast

            data_args = DataArguments()
            data_args.video_max_frames = 768
            data_args.video_min_frames = 16
            data_args.base_interval = 0.1
            data_args.max_pixels = 61250
            data_args.video_max_frame_pixels = 61250
            data_args.run_test = True
            data_args.image_processor = Qwen2VLImageProcessorFast.from_pretrained(model_path)
            data_args.audio_processor = WhisperFeatureExtractor(
                feature_size=data_args.feature_size, 
                sampling_rate=data_args.sampling_rate,
                hop_length=data_args.hop_length,
                chunk_length=data_args.chunk_length,
            )
            data_args.model_type = "qwen2.5vl"
            return data_args
        self.data_args = prepare_dataset(model_path)
        
        from qwenvl.data.dataset import make_supervised_data_module
        self.make_supervised_data_module = make_supervised_data_module

        print(f"[SUCCESS] {model_path} loaded.")

    def inference(self, item, max_frames):
        messages = {"video": item["video_path"],
                    "use_audio": True,
                    "conversations": [{"from": "human",
                                       "value": f"<video>\n{item['prompt']}"},
                                      {"from": "gpt", "value": ""}]}
        self.data_args.video_max_frames = max_frames

        data_module = self.make_supervised_data_module(tokenizer=self.tokenizer, data_args=self.data_args)
        test_data = data_module["train_dataset"]
        inputs = test_data._get_item(messages)
        inputs.pop("video", None)
        inputs.pop("image", None)
        inputs.pop("prompt", None)
        inputs.pop("ref", None)
        inputs.pop("audio", None)
        inputs.pop("use_audio", False)
        inputs.pop("should_use", True)
        inputs = {k: v.to(f"cuda:{torch.cuda.current_device()}") for k, v in inputs.items() if isinstance(v, torch.Tensor)}    

        # check input
        # input_len = inputs["input_ids"].shape[1]
        # print(f"[INFO] Duration: {item.get('duration', 'unknown')}")
        # print(f"[INFO] Video: {inputs['video_grid_thw']}")
        # print(f"[INFO] Input token: {input_len}")

        output = self.model.generate(**inputs,
                                     use_cache=True,
                                     do_sample=False,
                                     num_beams=1)
        response = self.tokenizer.decode(output[0, len(inputs["input_ids"][0]):], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return response


class OmniVinci:
    def __init__(self, model_path, load_audio_in_video=True, audio_length="max_3600"):
        from transformers import AutoProcessor, AutoModel

        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2"
        )
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        self.generation_config = self.model.default_generation_config
        self.generation_config.update(**{"max_length": 99999999})
        self.model.config.load_audio_in_video = load_audio_in_video
        self.processor.config.load_audio_in_video = load_audio_in_video
        self.model.config.audio_chunk_length = audio_length
        self.processor.config.audio_chunk_length = audio_length

        print(f"[SUCCESS] {model_path} loaded.")
        print("[INFO] Default parameters:")
        print(f"  load_audio_in_video = {load_audio_in_video}")
        print(f"  audio_length = {audio_length}")

    def inference(self, item, max_frames):
        messages = [{"role": "user",
                     "content": [{"type": "video", "video": item["video_path"]},
                                 {"type": "text", "text": item["prompt"]}]}]
        self.model.config.num_video_frames = max_frames
        self.processor.config.num_video_frames = max_frames

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = self.processor([text])
        inputs = inputs.to(self.model.device)

        # check input
        # input_len = inputs["input_ids"].shape[1]
        # print(f"[INFO] Duration: {item.get('duration', 'unknown')}")
        # print(f"[INFO] Ori Video: {inputs['media']['video'][0].shape}")
        # print(f"[INFO] Ori Audio: {inputs['media']['audio_info'][0][0]['new_audio_chunk_length']}")
        # print(f"[INFO] Input token: {input_len}")

        output = self.model.generate(input_ids=inputs.input_ids,
                                     media=getattr(inputs, "media", None),
                                     media_config=getattr(inputs, "media_config", None),
                                     generation_config=self.generation_config)
        response = self.processor.tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        return response


class MiniCPM_o_45:
    def __init__(self, model_path, generate_audio=False):
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
            init_vision=True,
            init_audio=True,
            init_tts=False
        )
        self.generate_audio = generate_audio
        if generate_audio:
            self.model.init_tts()

        print(f"[SUCCESS] {model_path} loaded.")
        print("[INFO] Default parameters:")
        print(f"  generate_audio = {generate_audio}")

    def inference(self, item, max_frames):
        os.environ["MAX_NUM_FRAMES"] = str(max_frames)

        messages = [{"role": "user",
                     "content": [{"type": "video_url", "video_url": {"url": item["video_path"],
                                                                     "use_audio": True}},
                                 {"type": "text", "text": item["prompt"]}]}]

        # check input
        # from minicpmo.utils import get_video_frame_audio_segments
        # video_frames, audio_segments, _ = get_video_frame_audio_segments(item["video_path"])
        # print(f"[INFO] Duration: {item.get('duration', 'unknown')}")
        # print(f"[INFO] num frames: {len(video_frames)}")
        # print(f"[INFO] Audio: {len(audio_segments)}")

        response = self.model.chat(
            msgs=messages,
            do_sample=False,
            use_tts_template=True,
            enable_thinking=False,
            omni_mode=True,  # Required for omni inference
            generate_audio=self.generate_audio,
            max_slice_nums=1,  # Increase for HD mode
            max_inp_length=16384
        )
        return response


class VITA_15:
    def __init__(self, model_path):
        from vita.constants import (
            IMAGE_TOKEN_INDEX,
            DEFAULT_AUDIO_TOKEN,
            DEFAULT_IMAGE_TOKEN,
        )
        from vita.util.mm_utils import (
            KeywordsStoppingCriteria,
            get_model_name_from_path,
            tokenizer_image_audio_token,
            tokenizer_image_token
        )
        from vita.model.builder import load_pretrained_model
        from vita.conversation import SeparatorStyle, conv_templates
        self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self.DEFAULT_AUDIO_TOKEN = DEFAULT_AUDIO_TOKEN
        self.DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        self.KeywordsStoppingCriteria = KeywordsStoppingCriteria
        self.tokenizer_image_audio_token = tokenizer_image_audio_token
        self.tokenizer_image_token = tokenizer_image_token
        self.SeparatorStyle = SeparatorStyle
        self.conv_templates = conv_templates

        self.conv_mode = "qwen2p5_instruct"

        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model_path, None, model_name, "qwen2p5_instruct"
        )

        audio_encoder = self.model.get_audio_encoder()
        audio_encoder.to(dtype=torch.float16)
        self.audio_processor = audio_encoder.audio_processor

        import numpy as np
        from PIL import Image
        from decord import VideoReader, cpu
        def _get_rawvideo_dec(
            video_path,
            image_processor,
            max_frames=16,
            min_frames=4,
            video_framerate=1,
            s=None,
            e=None,
            image_aspect_ratio="pad",
        ):
            # speed up video decode via decord.

            if s is None:
                start_time, end_time = None, None
            else:
                start_time = int(s)
                end_time = int(e)
                start_time = start_time if start_time >= 0.0 else 0.0
                end_time = end_time if end_time >= 0.0 else 0.0
                if start_time > end_time:
                    start_time, end_time = end_time, start_time
                elif start_time == end_time:
                    end_time = start_time + 1

            if os.path.exists(video_path):
                vreader = VideoReader(video_path, ctx=cpu(0))
            else:
                print(video_path)
                raise FileNotFoundError

            fps = vreader.get_avg_fps()
            f_start = 0 if start_time is None else int(start_time * fps)
            f_end = int(min(1000000000 if end_time is None else end_time * fps, len(vreader) - 1))
            num_frames = f_end - f_start + 1
            if num_frames > 0:
                # T x 3 x H x W
                sample_fps = int(video_framerate)
                t_stride = int(round(float(fps) / sample_fps))

                all_pos = list(range(f_start, f_end + 1, t_stride))
                if len(all_pos) > max_frames:
                    sample_pos = [
                        all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=max_frames, dtype=int)
                    ]
                elif len(all_pos) < min_frames:
                    sample_pos = [
                        all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=min_frames, dtype=int)
                    ]
                else:
                    sample_pos = all_pos

                patch_images = [Image.fromarray(f) for f in vreader.get_batch(sample_pos).asnumpy()]

                if image_aspect_ratio == "pad":

                    def expand2square(pil_img, background_color):
                        width, height = pil_img.size
                        if width == height:
                            return pil_img
                        elif width > height:
                            result = Image.new(pil_img.mode, (width, width), background_color)
                            result.paste(pil_img, (0, (width - height) // 2))
                            return result
                        else:
                            result = Image.new(pil_img.mode, (height, height), background_color)
                            result.paste(pil_img, ((height - width) // 2, 0))
                            return result

                    patch_images = [
                        expand2square(i, tuple(int(x * 255) for x in image_processor.image_mean))
                        for i in patch_images
                    ]
                    patch_images = [
                        image_processor.preprocess(i, return_tensors="pt")["pixel_values"][0]
                        for i in patch_images
                    ]
                else:
                    patch_images = [
                        image_processor.preprocess(i, return_tensors="pt")["pixel_values"][0]
                        for i in patch_images
                    ]

                patch_images = torch.stack(patch_images)
                slice_len = patch_images.shape[0]

                return patch_images, slice_len
            else:
                print("video path: {} error.".format(video_path))
        self._get_rawvideo_dec = _get_rawvideo_dec

        print(f"[SUCCESS] {model_path} loaded.")

    def inference(self, item, max_frames):
        # Add VITA/vita/model/multimodal_encoder/whale/init_model.py Line 45 for audio max_length
        """
        max_length = 120
        max_samples = max_length * sample_rate
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]
        """
        audio, audio_for_llm_lens = self.audio_processor.process(item["video_path"])
        audio_length = audio.shape[0]
        audio = torch.unsqueeze(audio, dim=0)
        audio_length = torch.unsqueeze(torch.tensor(audio_length), dim=0)
        audio_for_llm_lens = torch.unsqueeze(torch.tensor(audio_for_llm_lens), dim=0)
        audios = dict()
        audios["audios"] = audio.half().cuda()
        audios["lengths"] = audio_length.half().cuda()
        audios["lengths_for_llm"] = audio_for_llm_lens.cuda()

        video_frames, slice_len = self._get_rawvideo_dec(
            item["video_path"],
            self.image_processor,
            max_frames=max_frames,
            video_framerate=1,
            image_aspect_ratio=getattr(self.model.config, "image_aspect_ratio", None),
        )
        image_tensor = video_frames.half().cuda()

        qs = self.DEFAULT_IMAGE_TOKEN * slice_len + self.DEFAULT_AUDIO_TOKEN + "\n" + item["prompt"]

        conv = self.conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt("video")
        input_ids = (
            self.tokenizer_image_audio_token(prompt, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )
        stop_str = conv.sep if conv.sep_style != self.SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = self.KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        # check input
        # input_len = input_ids.shape[1] + image_tensor.shape[0] * 256 + audios["lengths_for_llm"]
        # print(f"[INFO] Duration: {item.get('duration', 'unknown')}")
        # print(f"[INFO] Video: {image_tensor.shape}")
        # print(f"[INFO] Audio: {audios['lengths_for_llm']}")  # duration * 100 / 8
        # print(f"[INFO] Input token: {input_len}")
        # if input_len >= 6200:
        #     raise ValueError(f"Input length {input_len} exceeds model limit. Skipping.")

        output = self.model.generate(
            input_ids,
            images=image_tensor,
            audios=audios,
            do_sample=False,
            num_beams=1,
            max_new_tokens=10,
            stopping_criteria=[stopping_criteria],
            shared_v_pid_stride=None
        )
        response = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        response = response.strip()
        if '\u261c' in response:
            response = response[1:]
        response = response.strip()
        return response


class VITA_15_sft:
    def __init__(self, model_path):
        from vita.constants import (
            IMAGE_TOKEN_INDEX,
            DEFAULT_AUDIO_TOKEN,
            DEFAULT_IMAGE_TOKEN,
        )
        from vita.util.mm_utils import (
            KeywordsStoppingCriteria,
            get_model_name_from_path,
            tokenizer_image_audio_token,
            tokenizer_image_token
        )
        from vita.model.builder import load_pretrained_model
        from vita.conversation import SeparatorStyle, conv_templates
        self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self.DEFAULT_AUDIO_TOKEN = DEFAULT_AUDIO_TOKEN
        self.DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        self.KeywordsStoppingCriteria = KeywordsStoppingCriteria
        self.tokenizer_image_audio_token = tokenizer_image_audio_token
        self.tokenizer_image_token = tokenizer_image_token
        self.SeparatorStyle = SeparatorStyle
        self.conv_templates = conv_templates

        self.conv_mode = "qwen2p5_instruct"

        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model_path, None, model_name, "qwen2p5_instruct"
        )

        audio_encoder = self.model.get_audio_encoder()
        audio_encoder.to(dtype=torch.float16)
        self.audio_processor = audio_encoder.audio_processor

        import numpy as np
        from PIL import Image
        from decord import VideoReader, cpu
        def _get_rawvideo_dec(
            video_path,
            image_processor,
            max_frames=16,
            min_frames=4,
            video_framerate=1,
            s=None,
            e=None,
            image_aspect_ratio="pad",
        ):
            # speed up video decode via decord.

            if s is None:
                start_time, end_time = None, None
            else:
                start_time = int(s)
                end_time = int(e)
                start_time = start_time if start_time >= 0.0 else 0.0
                end_time = end_time if end_time >= 0.0 else 0.0
                if start_time > end_time:
                    start_time, end_time = end_time, start_time
                elif start_time == end_time:
                    end_time = start_time + 1

            if os.path.exists(video_path):
                vreader = VideoReader(video_path, ctx=cpu(0))
            else:
                print(video_path)
                raise FileNotFoundError

            fps = vreader.get_avg_fps()
            f_start = 0 if start_time is None else int(start_time * fps)
            f_end = int(min(1000000000 if end_time is None else end_time * fps, len(vreader) - 1))
            num_frames = f_end - f_start + 1
            if num_frames > 0:
                # T x 3 x H x W
                sample_fps = int(video_framerate)
                t_stride = int(round(float(fps) / sample_fps))

                all_pos = list(range(f_start, f_end + 1, t_stride))
                if len(all_pos) > max_frames:
                    sample_pos = [
                        all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=max_frames, dtype=int)
                    ]
                elif len(all_pos) < min_frames:
                    sample_pos = [
                        all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=min_frames, dtype=int)
                    ]
                else:
                    sample_pos = all_pos

                patch_images = [Image.fromarray(f) for f in vreader.get_batch(sample_pos).asnumpy()]

                if image_aspect_ratio == "pad":

                    def expand2square(pil_img, background_color):
                        width, height = pil_img.size
                        if width == height:
                            return pil_img
                        elif width > height:
                            result = Image.new(pil_img.mode, (width, width), background_color)
                            result.paste(pil_img, (0, (width - height) // 2))
                            return result
                        else:
                            result = Image.new(pil_img.mode, (height, height), background_color)
                            result.paste(pil_img, ((height - width) // 2, 0))
                            return result

                    patch_images = [
                        expand2square(i, tuple(int(x * 255) for x in image_processor.image_mean))
                        for i in patch_images
                    ]
                    patch_images = [
                        image_processor.preprocess(i, return_tensors="pt")["pixel_values"][0]
                        for i in patch_images
                    ]
                else:
                    patch_images = [
                        image_processor.preprocess(i, return_tensors="pt")["pixel_values"][0]
                        for i in patch_images
                    ]

                patch_images = torch.stack(patch_images)
                slice_len = patch_images.shape[0]

                return patch_images, slice_len
            else:
                print("video path: {} error.".format(video_path))
        self._get_rawvideo_dec = _get_rawvideo_dec

        print(f"[SUCCESS] {model_path} loaded.")

    def inference(self, item, max_frames):
        # Modify VITA/vita/model/multimodal_encoder/whale/module/layer/attention.py Line 85 for max_len: int = 5000 -> 7500
        audio, audio_for_llm_lens = self.audio_processor.process(item["video_path"])
        audio_length = audio.shape[0]
        audio = torch.unsqueeze(audio, dim=0)
        audio_length = torch.unsqueeze(torch.tensor(audio_length), dim=0)
        audio_for_llm_lens = torch.unsqueeze(torch.tensor(audio_for_llm_lens), dim=0)
        audios = dict()
        audios["audios"] = audio.half().cuda()
        audios["lengths"] = audio_length.half().cuda()
        audios["lengths_for_llm"] = audio_for_llm_lens.cuda()

        video_frames, slice_len = self._get_rawvideo_dec(
            item["video_path"],
            self.image_processor,
            max_frames=max_frames,
            video_framerate=1,
            image_aspect_ratio=getattr(self.model.config, "image_aspect_ratio", None),
        )
        image_tensor = video_frames.half().cuda()

        qs = self.DEFAULT_IMAGE_TOKEN * slice_len + self.DEFAULT_AUDIO_TOKEN + "\n" + item["prompt"]

        conv = self.conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt("video")
        input_ids = (
            self.tokenizer_image_audio_token(prompt, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )
        stop_str = conv.sep if conv.sep_style != self.SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = self.KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        # check input
        # input_len = input_ids.shape[1] + image_tensor.shape[0] * 256 + audios["lengths_for_llm"]
        # print(f"[INFO] Duration: {item.get('duration', 'unknown')}")
        # print(f"[INFO] Video: {image_tensor.shape}")
        # print(f"[INFO] Audio: {audios['lengths_for_llm']}")  # duration * 100 / 8
        # print(f"[INFO] Input token: {input_len}")

        output = self.model.generate(
            input_ids,
            images=image_tensor,
            audios=audios,
            do_sample=False,
            num_beams=1,
            max_new_tokens=10,
            stopping_criteria=[stopping_criteria],
            shared_v_pid_stride=None
        )
        response = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        response = response.strip()
        if '\u261e' in response:
            response = response[1:]
        response = response.strip()
        return response
