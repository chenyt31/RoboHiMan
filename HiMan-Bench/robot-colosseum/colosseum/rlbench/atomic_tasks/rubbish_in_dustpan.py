from typing import List
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition


class RubbishInDustpan(Task):

    def init_task(self) -> None:
        success_sensor = ProximitySensor('success')
        self.rubbish = Shape('rubbish')
        self.register_graspable_objects([self.rubbish])
        self.register_success_conditions(
            [DetectedCondition(self.rubbish, success_sensor)])

    def init_episode(self, index: int) -> List[str]:
        formatted_desc = {
            "vanilla": [
                f"drop the rubbish into the dustpan"
            ],
            "oracle_half": [
                f"pick up the rubbish on the table\ndrop the rubbish into the dustpan"
            ],
            "oracle_full": [
                f"drop the rubbish into the dustpan"
            ]
        }
        return formatted_desc

    def variation_count(self) -> int:
        return 1