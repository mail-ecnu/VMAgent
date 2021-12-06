A2C
===

A2C [#]_ is a popular Actor-Critic reinforcement learning algorithm which uses the advantage function instead of the
original return in the critical network. In our VMAgent, we implement the A2C with advantage function. The criitic out
the q value for each server (NUMA) and the actor out probility of actions.

Example
-------

Train A2C in fading environment with 5 servers, and parameters gamma=0.99 learning_rate=0.003:

.. code:: bash

   python vmagent/train.py --env fading --alg a2c --N 5 --gamma 0.99 --lr 0.003

.. [#] Schulman J, Moritz P, Levine S, et al. High-dimensional continuous control using generalized advantage
   estimation[J]. arXiv preprint arXiv:1506.02438, 2015.
