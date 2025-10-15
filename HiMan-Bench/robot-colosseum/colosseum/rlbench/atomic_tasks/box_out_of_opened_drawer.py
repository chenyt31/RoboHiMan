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


class BoxOutOfOpenedDrawer(Task):

    def init_task(self) -> None:
        self._options = ['bottom', 'middle', 'top']
        self._anchors = [Dummy('waypoint_anchor_%s' % opt)
                         for opt in self._options]
        self._joints = [Joint('drawer_joint_%s' % opt)
                        for opt in self._options]
        self._block_pos = Dummy('block_pos')
        self._mark = Dummy('mark')
        self._item = Shape('strawberry_jello')
        self.register_graspable_objects([self._item])

    def init_episode(self, index: int) -> List[str]:
        option = self._options[index]
        anchor = self._anchors[index]
        self._mark.set_position(anchor.get_position())
        _, _, z_target = anchor.get_position()
        x, y, _ = self._block_pos.get_position()
        self._joints[index].set_joint_position(0.24)
        self._joints[(index+1)%3].set_joint_position(0.0)
        self._joints[(index+2)%3].set_joint_position(0.0)
        self._item.set_position([x, y, z_target])
        
        success_sensor = ProximitySensor('success')
        block_on = DetectedCondition(self._item, success_sensor)
        self.register_success_conditions([block_on])

        formatted_desc = {
            "vanilla": [
                f"take the strawberry jello out of the {option} drawer and place the strawberry jello on the drawer's surface"
            ],
            "oracle_half": [
                f"pick up the strawberry jello in the {option} drawer\nplace the strawberry jello on the drawer's surface"
            ],
            "oracle_full": [
                f"take the strawberry jello out of the {option} drawer and place the strawberry jello on the drawer's surface"
            ]
        }
        return formatted_desc

    def variation_count(self) -> int:
        return 3

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0, 0, - np.pi / 8], [0, 0, np.pi / 8]
