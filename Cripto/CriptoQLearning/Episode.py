# 1 episode =  96 steps
class Episode:

    # Each step is a list, ej.  step = [0.3, 0.2, 0.5]
    def __init__(self, steps):
        self.steps = steps

    @property
    def episode(self):
        return self.steps
