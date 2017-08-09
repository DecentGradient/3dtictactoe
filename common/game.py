import numpy as np
import random

class Game:


    def __init__(self,verbose=False):
        self.log = []
        self.setup()
        self.verbose = verbose
        self.wins = self.win_rows()
        return None

    def setup(self):
        self.last_move = None
        self.moves = 0
        self.skip = False
        self.space= np.zeros(27)
        self.game_over = False
        self.inv_game_over = False
        tmp = self.log
        self.log=[]
        return tmp


    def move(self,idx,val):
        if self.skip:
            self.skip = False
            print("skip 2")
            return None
        avail = self.avail()

        if idx not in avail:
            self.skip = True
            print("skip 1")
            return None
        self.log.append((idx, val))
        self.last_move = val
        self.space[idx] = val
        self.moves  +=1
        if self.verbose:
            print(idx)
            self.view()
        return self.log


    def check_row(self,row_tp):
        #print(self.space[row_tp[0]])
        if self.space[row_tp[0]] != 0.  and self.space[row_tp[0]] == self.space[row_tp[1]] and self.space[row_tp[1]] == self.space[row_tp[2]]:
            print(self.moves,row_tp,self.space[row_tp[0]],self.space[row_tp[1]],self.space[row_tp[2]])
            return True
        else:
            return False
    def win_rows(self):
        win_list = [
            # board 0 horizontal
            (0, 1, 2),
            (3, 4, 5),
            (6, 7, 8),
            # board 0 vertical
            (0, 3, 6),
            (1, 4, 7),
            (2, 5, 8),
            # board 0 diag
            (0, 4, 8),
            (6, 4, 2),

            # board0 horizontal
            (9, 10, 11),
            (12, 13, 14),
            (15, 16, 17),
            # board0 vertical
            (9, 12, 15),
            (10, 13, 16),
            (11, 14, 17),
            # board0 diag
            (9, 13, 17),
            (15, 13, 11),

            # board1 horizontal
            (18, 19, 20),
            (21, 22, 23),
            (24, 25, 26),
            # board1 vertical
            (18, 21, 24),
            (19, 22, 25),
            (20, 23, 26),
            # board1 diag
            (18, 22, 26),
            (24, 22, 20),

            # multi dimensional horizontal
            (0, 10, 20),
            (2, 10, 18),

            (3, 13, 23),
            (5, 13, 21),

            (6, 16, 26),
            (8, 16, 24),

            # multi dimentional vertical
            (0, 12, 24),
            (6, 12, 18),

            (1, 13, 25),
            (7, 13, 19),

            (2, 14, 26),
            (8, 14, 20),
            # multi dimentional diag
            (0, 13, 26),
            (8, 13, 18),
            (2, 13, 24),
            (6, 13, 20),
            # straight down the dimenstion
            (0, 9, 18),
            (1, 10, 19),
            (2, 11, 20),
            (3, 12, 21),
            (4, 13, 22),
            (5, 14, 23),
            (6, 15, 24),
            (7, 16, 25),
            (8, 17, 26)
        ]
        return win_list

    def check_wins(self):



        for row in self.wins:
            win = self.check_row(row)


            if win:
                self.game_over = 1 if self.space[row[0]] == 1 else 2
                self.inv_game_over =  2 if self.space[row[0]] == 1 else 1


    def avail(self):
        avail = [i for i, j in enumerate(self.space) if j == 0.]
        return avail

    def view(self):

        print(self.space[0], "|", self.space[1], "|", self.space[2])
        print(self.space[3], "|", self.space[4], "|", self.space[5])
        print(self.space[6], "|", self.space[7], "|", self.space[8])
        print("--------")
        print(self.space[9], "|", self.space[10], "|", self.space[11])
        print(self.space[12], "|", self.space[13], "|", self.space[14])
        print(self.space[15], "|", self.space[16], "|", self.space[17])
        print("--------")
        print(self.space[18], "|", self.space[19], "|", self.space[20])
        print(self.space[21], "|", self.space[22], "|", self.space[23])
        print(self.space[24], "|", self.space[25], "|", self.space[26])

        print("                    ")
        print(self.game_over, "xxxxxxxxxxxxxxxxxxxxxxxx")
        print("                             ")

        return None

    def inverse_space(self):
        inv = self.space
        for idx, val in enumerate(self.space):
            if val == 1:
                inv[idx] = 10
            elif val == 2 :
                inv[idx] = -10
        for idx,val in enumerate(inv):
            if val == 10:
                inv[idx] = 2
            elif val ==-10:
                inv[idx] = 1
        return inv

    def step(self,player):
        self.check_wins()

        state = self.space
        if self.game_over  and self.game_over==player:
            if self.moves>5:
                reward = 15 - (.2*self.moves)
            else:
                reward = 15
        elif self.game_over and self.game_over != player:
            reward = -10# + (.02*self.moves)
        else:
            reward = 0

        return (state,reward)
    def random_space(self):
        avail = self.avail()
        return random.choice(avail)

    def coord_space(self):

        return None