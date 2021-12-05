# DQN 
DQN[1] is a popular off-ploicy reinforcement learning algorithm.
In our VMAgent, we implement the DQN with Double Q[2] and Dueling Q[3].
The DQN agent out Q values for each server (NUMA) and we take epsilon-greedy to select action based on the Q values.

## Example
Train DQN in fading environment with 5 servers, and parameters gamma=0.99 learning_rate=0.003:
```
python vmagent/train.py --env fading --alg dqn --N 5 --gamma 0.99 --lr 0.003
```

[1] Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning." nature 518.7540 (2015): 529-533.
[2] Van Hasselt, Hado, Arthur Guez, and David Silver. "Deep reinforcement learning with double q-learning." Proceedings of the AAAI conference on artificial intelligence. Vol. 30. No. 1. 2016.
[3] Sewak, Mohit. "Deep q network (dqn), double dqn, and dueling dqn." Deep Reinforcement Learning. Springer, Singapore, 2019. 95-108.