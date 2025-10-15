import os
from jinja2 import Template

class PromptManager:
    def __init__(self, base_dir="prompts"):
        self.base_dir = base_dir

    def load_template(self, model_name: str, template_name: str):
        path = os.path.join(self.base_dir, model_name, template_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Prompt template not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return Template(f.read())

    def render(self, model_name: str, template_name: str, variables: dict):
        template: Template = self.load_template(model_name, template_name)
        return template.render(**variables)