from typing import List, Tuple

import numpy as np
from pyrep.objects.dummy import Dummy
from pyrep.objects.joint import Joint
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from rlbench.backend.conditions import DetectedCondition
from rlbench.backend.task import Task

BLOCK_NUM = 2

class TakeTwoOutOfDifferent(Task):

    def init_task(self) -> None:
        self._options = ['bottom', 'middle', 'top']
        self._anchors = [Dummy('waypoint_anchor_%s' % opt)
                         for opt in self._options]
        self._joints = [Joint('drawer_joint_%s' % opt)
                        for opt in self._options]
        self._mark0 = Dummy('mark0')
        self._mark1 = Dummy('mark1')
        self._blocks = [Shape('item' + str(i)) for i in range(BLOCK_NUM)]
        self.register_graspable_objects(self._blocks)
        self.index_comb = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]

    def init_episode(self, index: int) -> List[str]:
        idx0, idx1 = self.index_comb[index]
        option0 = self._options[idx0]
        anchor0 = self._anchors[idx0]
        self._mark0.set_position(anchor0.get_position())
        
        option1 = self._options[idx1]
        anchor1 = self._anchors[idx1]
        self._mark1.set_position(anchor1.get_position())
        
        _, _, z_target0 = anchor0.get_position()
        x, y, _ = self._blocks[0].get_position()
        self._blocks[0].set_position([x, y, z_target0])
        _, _, z_target1 = anchor1.get_position()
        x, y, _ = self._blocks[1].get_position()
        self._blocks[1].set_position([x, y, z_target1])

        success_sensor = ProximitySensor('success')
        block_conditions = [DetectedCondition(block, success_sensor) for block in self._blocks]
        self.register_success_conditions(block_conditions)

        formatted_desc = {
            "vanilla": [
                f"take one block out of the {option0} drawer and take the other block out of the {option1} drawer, then place them on the drawer's surface"
            ],
            "oracle_half": [
                f"grasp the {option0} drawer handle\npull the {option0} drawer open\npick up the block in the {option0} drawer\nplace the block on the drawer's surface\nmove close to the {option0} drawer handle\npush the {option0} drawer shut\ngrasp the {option1} drawer handle\npull the {option1} drawer open\npick up the block in the {option1} drawer\nplace the block on the drawer's surface"
            ],
            "oracle_full": [
                f"open the {option0} drawer\ntake the block out of the {option0} drawer and place the block on the drawer's surface\nclose the {option0} drawer\nopen the {option1} drawer\ntake the block out of the {option1} drawer and place the block on the drawer's surface"
            ]
        }
        return formatted_desc

    def variation_count(self) -> int:
        return 6

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0, 0, - np.pi / 8], [0, 0, np.pi / 8]
