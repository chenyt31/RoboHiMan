from typing import List, Tuple
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import Task
from pyrep.objects.object import Object

from rlbench.backend.conditions import DetectedCondition
from rlbench.backend.spawn_boundary import SpawnBoundary


class BroomOutOfCupboard(Task):

    def init_task(self) -> None:
        self.sample_dummy = Shape('boundary_root2')
        self.broom = Shape('broom')
        self.broom_holder = Shape('broom_holder')
        self.success_sensor = ProximitySensor('success')
        self.register_graspable_objects([self.broom])
        self.boundary = SpawnBoundary([Shape('workspace')])

    def init_episode(self, index: int) -> List[str]:
        self.boundary.clear()
        self.boundary.sample(self.sample_dummy, ignore_collisions=True)
        conditions = [DetectedCondition(self.broom, self.success_sensor)]
        self.register_success_conditions(conditions)
        
        formatted_desc = {
            "vanilla": [
                f"put the broom on the table"
            ],
            "oracle_half": [
                f"pick up the broom in the cupboard\nplace the broom on the table"
            ],
            "oracle_full": [
                f"put the broom on the table"
            ]
        }
        return formatted_desc

    def variation_count(self) -> int:
        return 1
    
    def boundary_root(self) -> Object:
        return Shape('boundary_root')

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0, 0, - np.pi / 8], [0, 0, np.pi / 8]
