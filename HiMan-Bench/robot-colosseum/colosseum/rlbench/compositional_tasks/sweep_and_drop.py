from typing import List
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition

DIRT_NUM = 5


class SweepAndDrop(Task):

    def init_task(self) -> None:
        self.broom = Shape('broom')
        self.success_sensor = ProximitySensor('success')
        self.dirts = [Shape('dirt' + str(i)) for i in range(DIRT_NUM)]
        self.rubbish = Shape('rubbish')
        self.register_graspable_objects([self.rubbish, self.broom])

    def init_episode(self, index: int) -> List[str]:
        sweep_conditions = [DetectedCondition(dirt, self.success_sensor) for dirt in self.dirts]
        rubbish_conditions = [DetectedCondition(self.rubbish, self.success_sensor)]
        self.register_success_conditions(sweep_conditions + rubbish_conditions)
                
        formatted_desc = {
            "vanilla": [
                f"clean all the dirt and rubbish into the dustpan"
            ],
            "oracle_half": [
                f"pick up the rubbish on the table\ndrop the rubbish into the dustpan\npick up the broom on the table\nsweep dirt to dustpan"
            ],
            "oracle_full": [
                f"drop the rubbish into the dustpan\nsweep dirt to dustpan"
            ]
        }
        return formatted_desc
    
    def variation_count(self) -> int:
        return 1
