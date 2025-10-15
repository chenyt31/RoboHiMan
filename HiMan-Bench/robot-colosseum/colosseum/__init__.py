import os

COLOSSEUM_ROOT = os.path.dirname(os.path.abspath(__file__))

TASKS_PY_FOLDER = os.path.join(COLOSSEUM_ROOT, "rlbench", "tasks")
TASKS_TTM_FOLDER = os.path.join(COLOSSEUM_ROOT, "rlbench", "task_ttms")

ATOMIC_TASKS_PY_FOLDER = os.path.join(COLOSSEUM_ROOT, "rlbench", "atomic_tasks")
ATOMIC_TASKS_TTM_FOLDER = os.path.join(COLOSSEUM_ROOT, "rlbench", "atomic_task_ttms")
COMPOSITIONAL_TASKS_PY_FOLDER = os.path.join(COLOSSEUM_ROOT, "rlbench", "compositional_tasks")
COMPOSITIONAL_TASKS_TTM_FOLDER = os.path.join(COLOSSEUM_ROOT, "rlbench", "compositional_task_ttms")

ASSETS_FOLDER = os.path.join(COLOSSEUM_ROOT, "assets")
ASSETS_TEXTURES_FOLDER = os.path.join(ASSETS_FOLDER, "textures")
ASSETS_MODELS_TTM_FOLDER = os.path.join(ASSETS_FOLDER, "models")
ASSETS_CONFIGS_FOLDER = os.path.join(ASSETS_FOLDER, "configs")
ASSETS_ATOMIC_CONFIGS_FOLDER = os.path.join(ASSETS_FOLDER, "atomic_configs")
ASSETS_COMPOSITIONAL_CONFIGS_FOLDER = os.path.join(ASSETS_FOLDER, "compositional_configs")
ASSETS_JSON_FOLDER = os.path.join(ASSETS_FOLDER, "json")
ASSETS_ATOMIC_JSON_FOLDER = os.path.join(ASSETS_FOLDER, "atomic_json")
ASSETS_COMPOSITIONAL_JSON_FOLDER = os.path.join(ASSETS_FOLDER, "compositional_json")
