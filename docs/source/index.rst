.. VMAgent documentation master file, created by
   sphinx-quickstart on Tue Nov 30 16:55:35 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to VMAgent's documentation!
===================================
.. figure:: ./images/logo.svg
    :width: 666px
    :align: center
    :alt: VMAgent
    :target: https://github.com/mail-ecnu/VMAgent

VMAgent is a platform for exploiting Reinforcement Learning (RL) on Virtual Machine (VM)Scheduling tasks.
It contains multiple practicle VM scheduling scenarios (such as Fading, Rcovering, etc).
These scenarios also correspond to the challanges in the RL.
Exploiting the design of RL methods in these secenarios help both the RL and VM scheduling community.

Key Components of VMAgent:

- SchedGym (Simulator): it provides many practical scenarios and flexible configurations to define custom scenarios.
- SchedAgent (Algorithms): it provides many popular RL methods as the baselines.
- SchedVis (Visulization): it provides the visualization of schedlueing dynamics on many metrics.

Contents
----------

.. toctree::
    :maxdepth: 2
    :caption: Installation

    installation/installation.md
    
.. toctree::
    :maxdepth: 2
    :caption: SchedGym
   
    simulator/scenarios.md
    simulator/gym.md
    
.. toctree::
    :maxdepth: 2
    :caption: SchedAgent

    algorithms/framework.md
    algorithms/dqn.md
    algorithms/a2c.md
    algorithms/ppo.md
    algorithms/sac.md
  
.. toctree::
    :maxdepth: 2
    :caption: SchedVis
    :hidden:

    visualization/usage.md 
    visualization/example.md 

.. toctree::
    :maxdepth: 2
    :caption: Examples

    examples/dqn4fading.md
    examples/sac4recovering.md

.. toctree::
    :maxdepth: 3
    :caption: API Documents
    :hidden:

    apidoc/schedgym/modules.rst
    apidoc/vmagent/modules.rst
