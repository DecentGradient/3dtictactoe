import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle as p


#ar = [i for i in range(10)]
#with open('data/ptest','wb') as f:
#    p.dump(ar,f)
#arr = np.load("data/test.npy")
with open('data/randpgtest','rb') as f :
    #arr = np.array([i for i in range(20)])
    #p.dump(arr,f)
    arr= p.load(f)

for i in range(len(arr)):
    #print(arr)
    #if len(arr) ==1:
    #df = pd.DataFrame(arr)
    #else:
    #print(arr)
    df = pd.DataFrame(arr[i])
    #print(df)
    #exit()
    winp = plt.plot(df[0],df[1],label="P1 Wins")
    comp = plt.plot(df[0],df[2],label="P1 Immediate Completion")
    bloc = plt.plot(df[0],df[3],label="P1 Immediate Block")
    winp2 = plt.plot(df[0],df[4],label="P2 Wins")
    comp2 = plt.plot(df[0],df[5],label="P2 Immediate Completion")
    bloc2 = plt.plot(df[0],df[6],label="P2 Immediate Block")
    plt.legend()
    plt.ylim(0,1)
    plt.title("DDPG Deep Stochastic vs DDPG Deep")
    plt.ylabel('Probability')
    plt.xlabel("Episode")
    plt.show()
    print(winp)
