# Framework 
Our VMAgent's framework is mainly based on the [pymarl](https://github.com/oxwhirl/pymarl).
It consists `controller`, `learner`, `components`, `modules` and `utils`.

## Controller
The controller plays the role on output actions for sampling.

## Learner 
The learner plays is to update the agent's policy 

## Components
It provides key components for learning and sampling.
It includes the implementations of replay memory and action selector (i.e., epsilon-greedy action selection).

## Modules
It provides different networks of the agent, including critic network, policy network etc.

## Utils
It provides several utils for reinforcement learning.