from typing import List, Tuple
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.objects.object import Object
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition, NothingGrasped
from rlbench.backend.spawn_boundary import SpawnBoundary

GROCERY_NAMES = [
    'strawberry jello',
    'spam',
    'sugar',
]

class BoxInCupboard(Task):

    def init_task(self) -> None:
        self.groceries = [Shape(name.replace(' ', '_'))
                          for name in GROCERY_NAMES]
        self.grasp_points = [Dummy('%s_grasp_point' % name.replace(' ', '_'))
                             for name in GROCERY_NAMES]
        self.put_points = [Dummy('%s_point' % name.replace(' ', '_'))
                        for name in GROCERY_NAMES]
        self._waypoint1 = Dummy('waypoint1')
        self._waypoint5 = Dummy('waypoint5')
        self.register_graspable_objects(self.groceries)
        self.boundary = SpawnBoundary([Shape('workspace')])

    def init_episode(self, index: int) -> List[str]:
        self.boundary.clear()
        [self.boundary.sample(g, min_distance=0.1) for g in self.groceries]
        self._waypoint1.set_pose(self.grasp_points[index].get_pose())
        self._waypoint5.set_pose(self.put_points[index].get_pose())
        self.register_success_conditions([DetectedCondition(self.groceries[index], ProximitySensor('success'))])
        
        option = GROCERY_NAMES[index]
        formatted_desc = {
            "vanilla": [
                f"put the {option} in the cupboard"
            ],
            "oracle_half": [
                f"pick up the {option} on the table\nplace the {option} in the cupboard"
            ],
            "oracle_full": [
                f"put the {option} in the cupboard"
            ]
        }
        return formatted_desc

    def variation_count(self) -> int:
        return len(GROCERY_NAMES)

    def boundary_root(self) -> Object:
        return Shape('boundary_root')

    def base_rotation_bounds(self) -> Tuple[Tuple[float, float, float],
                                            Tuple[float, float, float]]:
        return (0.0, 0.0, -1.), (0.0, 0.0, 1.)

