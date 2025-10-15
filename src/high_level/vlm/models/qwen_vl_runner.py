import os
import json
import torch
from openai import OpenAI
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from src.high_level.vlm.utils.xml_patch_parser import extract_xml_fields
from src.high_level.vlm.utils.vision_process import process_vision_info


class QwenVLRunner:
    def __init__(self, mode="local", model_name="Qwen/Qwen2.5-VL-7B-Instruct",
                 ip="0.0.0.0", port=8000, device="cuda"):
        """
        Runner class to interact with Qwen-VL model either via:
        - "api" (through OpenAI-compatible API endpoint)
        - "local" (directly load model and processor in this environment)

        Args:
            mode (str): "api" or "local"
            model_name (str): HuggingFace model name or API model name
            ip (str): server IP if using API mode
            port (int): server port if using API mode
            device (str): device placement, e.g. "cuda" or "cpu"
        """
        self.mode = mode
        self.device = device

        if mode == "api":
            # Initialize an OpenAI-compatible client
            self.client = OpenAI(
                api_key=os.getenv("API_KEY", "0"),
                base_url=f"http://{ip}:{port}/v1"
            )
            self.model_name = model_name

        elif mode == "local":
            # Load model and processor locally
            self.model_name = model_name
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=torch.bfloat16, device_map=device
            )
            self.processor = AutoProcessor.from_pretrained(model_name, device_map=device)

            # Ensure left-padding for chat-style generation
            self.processor.tokenizer.padding_side = "left"

        else:
            raise ValueError("mode must be 'api' or 'local'")

    def prepare_messages(self, rendered_prompt: str):
        """
        Convert rendered JSON string into OpenAI-style messages.

        Args:
            rendered_prompt (str): JSON-encoded message content (text + multimodal info)

        Returns:
            list: messages formatted with system + user roles
        """
        content = json.loads(rendered_prompt)
        return [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {"role": "user", "content": content}
        ]

    def infer(self, rendered_prompt: str | list[str]):
        """
        Run inference with the Qwen-VL model.

        Args:
            rendered_prompt (str | list[str]): JSON-encoded message content
                - str: single prompt
                - list[str]: batch of prompts

        Returns:
            list[str]: decoded model outputs (length = batch size)
        """
        # --- Normalize input into a list ---
        is_single = isinstance(rendered_prompt, str)
        if is_single:
            rendered_prompt = [rendered_prompt]

        # --- Prepare messages for each prompt ---
        messages_batch = [self.prepare_messages(p) for p in rendered_prompt]

        if self.mode == "api":
            # API mode: run one by one (OpenAI API does not support batch natively)
            outputs = []
            for messages in messages_batch:
                result = self.client.chat.completions.create(
                    messages=messages, model=self.model_name
                )
                outputs.append(result.choices[0].message.content)
            return outputs

        elif self.mode == "local":
            # Local mode: batch supported
            texts = [
                self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                for messages in messages_batch
            ]

            # Extract image/video inputs for each message
            vision_inputs = [process_vision_info(messages) for messages in messages_batch]
            image_inputs = [vi[0] for vi in vision_inputs]
            video_inputs = [vi[1] for vi in vision_inputs]

            # Normalize modality inputs: avoid passing lists that contain None
            def _normalize_modal_inputs(batch_list):
                if all(x is None for x in batch_list):
                    return None
                # Replace per-sample None with empty list so the processor won't fail
                return [([] if x is None else x) for x in batch_list]

            image_inputs = _normalize_modal_inputs(image_inputs)
            video_inputs = _normalize_modal_inputs(video_inputs)

            # Preprocess into model inputs
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            # Generate output tokens
            generated_ids = self.model.generate(**inputs, max_new_tokens=256)

            # Slice out only the generated part
            out_ids_trimmed = generated_ids[:, inputs.input_ids.shape[1]:]

            # Decode batch into strings
            outputs = self.processor.batch_decode(out_ids_trimmed, skip_special_tokens=True)
            return outputs

    def parse_vlm_response(self, text: str, fields=("reasoning", "sub_task")):
        """
        Parse structured response fields from VLM output.

        Args:
            text (str): raw text response from the model
            fields (tuple[str]): fields to extract (e.g. reasoning, sub_task)

        Returns:
            dict: extracted field values
        """
        return extract_xml_fields(text, fields=fields)