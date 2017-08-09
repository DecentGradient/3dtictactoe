from common.benchmark import GameRate

from common.game import Game
from collections import Counter

game = Game(verbose=False)
game.setup()
logs = []
wins = []
test_episodes = 1000
for i in range(test_episodes):
    print(i)

    while not game.game_over:

        move = game.random_space()
        game.move(move,1)

        #print(game.space)
        game.step()
        if not game.game_over:
            move = game.random_space()
            game.move(move, 2)
            game.step()
    wins.append(game.game_over)
    log = game.setup()
    logs.append(log)




r = GameRate(verbose=False, list=logs)

r.check_games(1)

c = Counter(wins)

# print("1", c[1])
# print("2", c[2])

print("win percentage",c[1]/(test_episodes-1))
print("immediate completions", r.completions / (r.completions + r.missed_completions))