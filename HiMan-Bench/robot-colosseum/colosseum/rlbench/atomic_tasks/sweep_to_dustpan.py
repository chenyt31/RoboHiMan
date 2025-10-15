from typing import List
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition
from rlbench.backend.spawn_boundary import SpawnBoundary

DIRT_NUM = 5


class SweepToDustpan(Task):

    def init_task(self) -> None:
        self.sample_dummy = Shape('boundary_root2')
        broom = Shape('broom')
        success_sensor = ProximitySensor('success')
        dirts = [Shape('dirt' + str(i)) for i in range(DIRT_NUM)]
        conditions = [DetectedCondition(dirt, success_sensor) for dirt in dirts]
        self.register_graspable_objects([broom])
        self.register_success_conditions(conditions)
        self.boundary = SpawnBoundary([Shape('workspace')])

    def init_episode(self, index: int) -> List[str]:
        self.boundary.clear()
        self.boundary.sample(self.sample_dummy, ignore_collisions=True, min_distance=0.01)
        
        formatted_desc = {
            "vanilla": [
                f"sweep dirt to dustpan"
            ],
            "oracle_half": [
                f"pick up the broom on the table\nsweep dirt to dustpan"
            ],
            "oracle_full": [
                f"sweep dirt to dustpan"
            ]
        }
        return formatted_desc

    def variation_count(self) -> int:
        return 1
