# DQN 
DQN[1] is a popular off-ploicy reinforcement learning algorithm.
In our VMAgent, we implement the DQN with Double Q[2] and Dueling Q[3].
The DQN agent out Q values for each server (NUMA) and we take epsilon-greedy to select action based on the Q values.

## Example
Train DQN in fading environment with 5 servers, and parameters gamma=0.99 learning_rate=0.003:
```
python vmagent/train.py --env fading --alg dqn --N 5 --gamma 0.99 --lr 0.003
```