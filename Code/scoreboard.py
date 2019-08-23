

class Scoreboard():
    def __init__(self, sort_param_idx=0, name='NoName'):
        self.sb = []
        self.spi = sort_param_idx
        self.name = name

    def addItem(self, param_list):
        if self.sb:
            assert (len(param_list) == len(self.sb[0]))
        self.sb.append(param_list)

    def print_scoreboard(self, k, key):
        assert (len(key) == len(self.sb[0]))
        self.sb.sort(key=lambda x: x[self.spi])

        print('=' * 20)
        print('Printing Scoreboard for', self.name)
        print('=' * 20)
        print("Top-{}".format(k))
        print('=' * 20)
        for i in range(k):
            for idx in range(len(key)):
                print('{}: {}'.format(key[idx], self.sb[i][idx]))
            print('\n')
        print('=' * 20)
        print("Last-{}".format(k))
        print('=' * 20)
        for i in range(-k, 0, 1):
            for idx in range(len(key)):
                print('{}: {}'.format(key[idx], self.sb[i][idx]))
            print('\n')

    def flush(self):
        self.sb = []
