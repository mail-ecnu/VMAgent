# PPO
PPO[1] is a popular Actor-Critic reinforcement learning algorithm which forces the updata of the policy not to be large.
In our VMAgent, we implement the PPO2.
The criitic out the q value for each server (NUMA) and the actor out probility of actions.

## Example
Train PPO in fading environment with 5 servers, and parameters gamma=0.99 learning_rate=0.003:
```
python vmagent/train.py --env fading --alg ppo --N 5 --gamma 0.99 --lr 0.003
```

[1] Schulman J, Wolski F, Dhariwal P, et al. Proximal policy optimization algorithms[J]. arXiv preprint arXiv:1707.06347, 2017.

