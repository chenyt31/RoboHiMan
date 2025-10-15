from typing import List, Tuple
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.objects.object import Object
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition
from rlbench.backend.spawn_boundary import SpawnBoundary

GROCERY_NAMES = [
    'sugar',
    'spam'
]


class BoxExchange(Task):

    def init_task(self) -> None:
        self.groceries = [Shape(name.replace(' ', '_'))
                          for name in GROCERY_NAMES]
        self.put_points = [Dummy('%s_point' % name.replace(' ', '_'))
                        for name in GROCERY_NAMES]
        self.waypoint8 = Dummy('waypoint8')
        self.register_graspable_objects(self.groceries)
        self.boundary = SpawnBoundary([Shape('workspace1')])

    def init_episode(self, index: int) -> List[str]:
        self.boundary.clear()
        [self.boundary.sample(Shape('spam'), min_distance=0.1)]
        self.waypoint8.set_pose(Dummy('spam_grasp_point').get_pose())
        self.register_success_conditions(
            [DetectedCondition(Shape('sugar'), ProximitySensor('success_ground')),
             DetectedCondition(Shape('spam'), ProximitySensor('success_cupboard'))])
        
        formatted_desc = {
            "vanilla": [
                f"exchange the positions of the spam and the sugar(put the sugar on the table and put the spam in the cupboard)"
            ],
            "oracle_half": [
                f"pick up the sugar in the cupboard\nplace the sugar on the table\npick up the spam on the table\nplace the spam in the cupboard"
            ],
            "oracle_full": [
                f"put the sugar on the table\nput the spam in the cupboard"
            ]
        }
        return formatted_desc

    def variation_count(self) -> int:
        return 1

    def boundary_root(self) -> Object:
        return Shape('boundary_root')

    def base_rotation_bounds(self) -> Tuple[Tuple[float, float, float],
                                            Tuple[float, float, float]]:
        return (0.0, 0.0, -1.), (0.0, 0.0, 1.)

