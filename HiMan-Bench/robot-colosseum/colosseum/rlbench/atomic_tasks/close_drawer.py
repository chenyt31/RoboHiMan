from typing import List, Tuple
import numpy as np
from pyrep.objects.dummy import Dummy
from pyrep.objects.joint import Joint
from colosseum.rlbench.extensions.conditions import DrawerCondition
from rlbench.backend.task import Task


class CloseDrawer(Task):

    def init_task(self) -> None:
        self._options = ['bottom', 'middle', 'top']
        self._anchors = [Dummy('waypoint_anchor_%s' % opt)
                         for opt in self._options]
        self._joints = [Joint('drawer_joint_%s' % opt)
                        for opt in self._options]
        self._mark = Dummy('mark')

    def init_episode(self, index: int) -> List[str]:
        option = self._options[index]
        self._mark.set_position(self._anchors[index].get_position())
        self._joints[index].set_joint_position(0.24)
        self._joints[(index+1)%3].set_joint_position(0.0)
        self._joints[(index+2)%3].set_joint_position(0.0)
        self.register_success_conditions(
            [DrawerCondition(self._joints[index], 0.03, "close")])
        
        formatted_desc = {
            "vanilla": [
                f"close the {option} drawer"
            ],
            "oracle_half": [
                f"move close to the {option} drawer handle\npush the {option} drawer shut"
            ],
            "oracle_full": [
                f"close the {option} drawer"
            ]
        }
        return formatted_desc

    def variation_count(self) -> int:
        return 3

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0, 0, - np.pi / 8], [0, 0, np.pi / 8]
