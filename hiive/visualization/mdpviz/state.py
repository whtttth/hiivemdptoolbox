class State(object):
    def __init__(self, name, index, terminal_state=False):
        if name is None or len(name.strip()) == 0:
            print()
        self.name = name
        self.index = index
        self.terminal_state = terminal_state

    def __repr__(self):
        # return self.name
        return 'State(%s, %s, %s)' % (self.name, self.index, self.terminal_state)

    def __str__(self):  # S{self.index}_
        ret = f'{self.name}'.replace(' ', '_').replace(':', '_').replace(',', '_')
        print(f'__str__(State): {ret}')
        return ret
        return self.name

    def __hash__(self):
        return self.__str__().__hash__()