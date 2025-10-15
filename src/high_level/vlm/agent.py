from src.high_level.vlm.prompt_manager import PromptManager
from src.high_level.vlm.models.qwen_vl_runner import QwenVLRunner

class MultiModalAgent:
    def __init__(self, runner: QwenVLRunner, base_dir: str, prompt_model_name: str):
        self.prompt_manager = PromptManager(base_dir=base_dir)
        self.runner = runner
        self.prompt_model_name = prompt_model_name

    def run(self, task_name, variables, fields=("reasoning", "sub_task")):
        """
        Run a single prompt through the VLM model.
        Returns a list[dict] (length = 1).
        """
        return self.run_batch(task_name, [variables], fields=fields)

    def run_batch(self, task_name, variables_list, fields=("reasoning", "sub_task")):
        """
        Run a batch of prompts through the VLM model.

        Args:
            task_name (str): template prefix, e.g. "next_action"
            variables_list (list[dict]): list of variable dicts for template rendering
            fields (tuple[str]): which fields to parse from model response

        Returns:
            list[dict]: list of parsed VLM outputs (one per input)
        """
        template_file = f"{task_name}_{self.runner.mode}.jinja2"

        # Render prompts for the whole batch
        rendered_prompts = [
            self.prompt_manager.render(self.prompt_model_name, template_file, vars_)
            for vars_ in variables_list
        ]

        # Run inference (batch supported)
        outputs = self.runner.infer(rendered_prompts)

        # Parse outputs
        return [self.runner.parse_vlm_response(text=o, fields=fields) for o in outputs]