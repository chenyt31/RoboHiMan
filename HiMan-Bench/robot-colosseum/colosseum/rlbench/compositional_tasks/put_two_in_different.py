from typing import List, Tuple

import numpy as np
from pyrep.objects.dummy import Dummy
from pyrep.objects.joint import Joint
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from rlbench.backend.conditions import DetectedCondition, OrConditions
from rlbench.backend.task import Task
from colosseum.rlbench.extensions.conditions import AndCondition

BLOCK_NUM = 2

class PutTwoInDifferent(Task):

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

    def init_episode(self, index) -> List[str]:
        idx0, idx1 = self.index_comb[index]
        option0 = self._options[idx0]
        anchor0 = self._anchors[idx0]
        self._mark0.set_position(anchor0.get_position())
        
        option1 = self._options[idx1]
        anchor1 = self._anchors[idx1]
        self._mark1.set_position(anchor1.get_position())
        
        success_sensor0 = ProximitySensor('success_' + option0)
        success_sensor1 = ProximitySensor('success_' + option1)

        condition0 = AndCondition([
            DetectedCondition(self._blocks[0], success_sensor0),
            DetectedCondition(self._blocks[1], success_sensor1)
        ])
        condition1 = AndCondition([
            DetectedCondition(self._blocks[0], success_sensor1),
            DetectedCondition(self._blocks[1], success_sensor0)
        ])

        or_condition = OrConditions([condition0, condition1])
        self.register_success_conditions([or_condition])
        
        formatted_desc = {
            "vanilla": [
                f"put one block in the {option0} drawer and put the other block in the {option1} drawer"
            ],
            "oracle_half": [
                f"grasp the {option0} drawer handle\npull the {option0} drawer open\npick up the block on the drawer's surface\nplace the block in the {option0} drawer\nmove close to the {option0} drawer handle\npush the {option0} drawer shut\ngrasp the {option1} drawer handle\npull the {option1} drawer open\npick up the block on the drawer's surface\nplace the block in the {option1} drawer"
            ],
            "oracle_full": [
                f"open the {option0} drawer\nput the block in the {option0} drawer\nclose the {option0} drawer\nopen the {option1} drawer\nput the block in the {option1} drawer"
            ]
        }
        return formatted_desc

    def variation_count(self) -> int:
        return 6

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0, 0, - np.pi / 8], [0, 0, np.pi / 8]
