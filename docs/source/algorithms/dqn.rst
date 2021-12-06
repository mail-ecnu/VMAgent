DQN
===

DQN [#]_ is a popular off-ploicy reinforcement learning algorithm. In our VMAgent, we implement the DQN with Double Q [#]_
and Dueling Q [#]_. The DQN agent out Q values for each server (NUMA) and we take epsilon-greedy to select action based on
the Q values.

Example
-------

Train DQN in fading environment with 5 servers, and parameters gamma=0.99 learning_rate=0.003:

.. code:: bash

   python vmagent/train.py --env fading --alg dqn --N 5 --gamma 0.99 --lr 0.003

.. [#] Mnih, Volodymyr, et al. “Human-level control through deep reinforcement learning.” nature 518.7540 (2015): 529-533.

.. [#] Van Hasselt, Hado, Arthur Guez, and David Silver. “Deep reinforcement learning with double q-learning.” Proceedings
   of the AAAI conference on artificial intelligence. Vol. 30. No. 1. 2016.

.. [#] Sewak, Mohit. “Deep q network (dqn), double dqn, and dueling dqn.” Deep Reinforcement Learning. Springer, Singapore,
   2019. 95-108.
