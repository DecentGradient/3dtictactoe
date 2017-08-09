from collections import Counter

from common.game import Game


class GameRate(Game):
    def __init__(self,verbose=False,list=None,player=1,opponent=2):
        super().__init__(verbose)
        self.games = list
        self.missed_completions = 0
        self.completions = 0
        self.blocks = 0
        self.missed_blocks = 0
        self.pl = player
        self.op = opponent

        return None

    def check_games(self):

        for game in self.games:

            self.new_two = []
            self.old_two = []
            self.new_op_two = []
            self.old_op_two = []
            self.replay(game)
            self.setup()


    def replay(self,game):

        for move,val in game:

            self.move(move,val)
            self.check_game()
            self.check_wins()
            if self.game_over:
                continue

    def check_game(self):
        won = False
        missed =0
        justmissed =False

        if self.pl != self.last_move:
            for idx, row in enumerate(self.wins):
                if self.two_in_row(row, self.op) and idx not in self.old_op_two:
                    self.new_op_two.append(idx)
            return None
        for idx in self.new_two:

            if self.three_in_row(self.wins[idx]):
                won = True
                if idx in self.new_two:
                    self.completions+=1
                self.new_two=[]
                break
            else:
                self.old_two.append(idx)
                self.new_two.remove(idx)
                if not self.unmissed(self.wins[idx]):

                    self.missed_completions +=1
                    justmissed = True



        if not won:
            for idx in self.new_op_two:

                if self.unblocked(self.wins[idx]):
                   self.missed_blocks+=1
                else:
                    self.old_op_two.append(idx)
                    self.new_op_two.remove(idx)
                    self.blocks += 1
                    if justmissed:
                        self.missed_completions-=1

        for idx,row in enumerate(self.wins):
            if self.two_in_row(row,self.pl) and idx not in self.old_two:
                self.new_two.append((idx))



    def two_in_row(self,row_tp,pl):

        c = Counter([self.space[row_tp[0]],self.space[row_tp[1]],self.space[row_tp[2]]])
        for num, count in c.items():
            if count == 2 and num == pl:
                return True
        return False
    def three_in_row(self,row_tp):

        c = Counter([self.space[row_tp[0]],self.space[row_tp[1]],self.space[row_tp[2]]])
        for num, count in c.items():
            if count == 3 and num == self.pl:
                return True
        return False
    def unblocked(self,row_tp):
        c = Counter([self.space[row_tp[0]], self.space[row_tp[1]], self.space[row_tp[2]]])
        if c[self.op]==3 or (c[self.op]==2 and c[self.pl]==0):
            return True
        else:
            return False

    def unmissed(self,row_tp):
        c = Counter([self.space[row_tp[0]], self.space[row_tp[1]], self.space[row_tp[2]]])
        if c[self.op] == 1 :
            return True
        else:
            return False