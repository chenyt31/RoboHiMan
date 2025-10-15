from typing import List, Tuple

import numpy as np
from pyrep.objects.dummy import Dummy
from pyrep.objects.joint import Joint
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from pyrep.objects.object import Object
from rlbench.backend.conditions import DetectedCondition
from rlbench.backend.task import Task
from rlbench.backend.spawn_boundary import SpawnBoundary


class TransferBox(Task):

    def init_task(self) -> None:
        self.sample_dummy = Shape('boundary_root2')

        self._options = ['bottom', 'middle', 'top']
        self._anchors = [Dummy('waypoint_anchor_%s' % opt)
                         for opt in self._options]
        self._joints = [Joint('drawer_joint_%s' % opt)
                        for opt in self._options]
        self._mark = Dummy('mark')
        self._item = Shape('strawberry_jello')
        self.register_graspable_objects([self._item])
        self.boundary = SpawnBoundary([Shape('workspace')])

    def init_episode(self, index: int) -> List[str]:
        self.boundary.clear()
        self.boundary.sample(self.sample_dummy, min_rotation=(0.0, 0.0, - np.pi / 8), max_rotation=(0.0, 0.0, np.pi / 8), ignore_collisions=True)
        option = self._options[index]
        anchor = self._anchors[index]
        self._mark.set_position(anchor.get_position())
        _, _, z_target = anchor.get_position()
        x, y, z = self._item.get_position()
        self._item.set_position([x, y, z_target])

        success_sensor = ProximitySensor('success')
        block_on = DetectedCondition(self._item, success_sensor)
        self.register_success_conditions([block_on])

        formatted_desc = {
            "vanilla": [
                f"take the strawberry jello out of the {option} drawer and put the strawberry jello in the cupboard"
            ],
            "oracle_half": [
                f"grasp the {option} drawer handle\npull the {option} drawer open\npick up the strawberry jello in the {option} drawer\nplace the strawberry jello in the cupboard"
            ],
            "oracle_full": [
                f"open the {option} drawer\ntake the strawberry jello out of the {option} drawer and place the strawberry jello on the drawer's surface\nput the strawberry jello in the cupboard"
            ]
        }
        return formatted_desc

    def variation_count(self) -> int:
        return 3

    def boundary_root(self) -> Object:
        return Shape('boundary_root')
    
    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0, 0, - np.pi / 8], [0, 0, np.pi / 8]
