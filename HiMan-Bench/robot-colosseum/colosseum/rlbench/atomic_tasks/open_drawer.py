from typing import List, Tuple

import numpy as np
from pyrep.objects.dummy import Dummy
from pyrep.objects.joint import Joint
# from rlbench.backend.conditions import DrawerCondition
from colosseum.rlbench.extensions.conditions import DrawerCondition
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from rlbench.backend.conditions import DetectedCondition
from rlbench.backend.task import Task


class OpenDrawer(Task):

    def init_task(self) -> None:
        self._options = ['bottom', 'middle', 'top']
        self._anchors = [Dummy('waypoint_anchor_%s' % opt)
                         for opt in self._options]
        self._joints = [Joint('drawer_joint_%s' % opt)
                        for opt in self._options]
        self._mark = Dummy('mark')

    def init_episode(self, index) -> List[str]:
        option = self._options[index]
        self._mark.set_position(self._anchors[index].get_position())
        self.register_success_conditions(
            [DrawerCondition(self._joints[index], 0.15, "open")])
        formatted_desc = {
            "vanilla": [
                f"open the {option} drawer"
            ],
            "oracle_half": [
                f"grasp the {option} drawer handle\npull the {option} drawer open"
            ],
            "oracle_full": [
                f"open the {option} drawer"
            ]
        }
        return formatted_desc

    def variation_count(self) -> int:
        return 3

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0, 0, - np.pi / 8], [0, 0, np.pi / 8]
