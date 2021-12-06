SAC
===

SAC [#]_ is a popular Actor-Critic reinforcement learning algorithm whith maximum entropy RL. In our VMAgent, we implement
the SAC with automatic entropy adjustment. The criitic out the q value for each server (NUMA) and the actor out
probility of actions.

Example
-------

Train SAC in fading environment with 5 servers, and parameters gamma=0.99 learning_rate=0.003:

.. code:: bash

   python vmagent/train.py --env fading --alg sac --N 5 --gamma 0.99 --lr 0.003

.. [#] Haarnoja T, Zhou A, Abbeel P, et al. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with
   a stochastic actor[C]//International conference on machine learning. PMLR, 2018: 1861-1870.
