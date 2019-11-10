class Sample:
    instructions = []
    opt = None
    compiler = None

    def __init__(self, values):
        self.__dict__ = values

    def no_parameters_instructions(self):
        """
        Keep only inst_name
        :return: a list of inst_names
        """
        res = []
        for instruction in self.instructions:
            res.append((str(instruction).split(' '))[0])
        return res

    def instructions_as_string(self):
        """
        Keep all instructions properties, removing special characters
        :return: singular-occurrence of instructions
        """
        modified = set()
        for instruction in self.instructions:
            x = instruction.replace('[', '').replace(']', '').replace('+', '').split(' ')
            for s in x:
                modified.add(s)
        return ' '.join(modified)
