import numpy as np

# This file is mostly for testing
# Does not handle bounds yet
# Should eventually be moved into the simulator

class Path:
    def __init__(self, directions: np.ndarray):
        self.directions = directions
        self.length = directions.shape[0]
        self.current_step = 0
    
    def move(self):
        if self.current_step >= self.length:
            return None
        dir = self.directions[self.current_step]
        self.current_step += 1
        return dir


class StairStepPath(Path):
    def __init__(self, step_length: int, num_steps: int):
        directions = []
        for _ in range(num_steps):
            for _ in range(step_length):
                directions.append([1, 0, 0])  # Move in x
            for _ in range(step_length):
                directions.append([0, 0, -1])  # Move in z

        super().__init__(np.array(directions, dtype=np.int32))