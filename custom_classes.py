class Sample:
    instructions = []
    opt = None
    compiler = None

    def __init__(self, values):
        self.__dict__ = values

    def no_parameters_instructions(self):
        res = []
        for instruction in self.instructions:
            x = (str(instruction).split(' '))[0]
            res.append(x)
        return res

