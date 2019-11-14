class Action(object):
    def __init__(self, name, index):
        self.name = name
        self.index = index

    def __repr__(self):
        return 'Action(%s, %s)' % (self.name, self.index)

    def __str__(self):  # A{self.index}_
        return f'{self.name}'.replace(' ', '_')

    def __hash__(self):
        return self.__str__().__hash__()