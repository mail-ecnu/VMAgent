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

VMAgent is a platform for exploiting Reinforcement Learning (RL) on Virtual Machine (VM) scheduling tasks.
It is developed by the Multi-Agent Artificial Intelligence Lab (MAIL) in East China Normal University and Algorithm Innovation Lab in Huawei Cloud.
VMAgent is constructed based on one month real VM scheduling dataset called `Huawei-East-1 <https://github.com/mail-ecnu/VMAgent/blob/master/vmagent/data/dataset.csv>`_ from HUAWEI Cloud and it contains multiple practicle VM scheduling scenarios (such as Fading, Rcovering, etc).
These scenarios also correspond to the challanges in the RL. Exploiting the design of RL methods in these secenarios help both the RL and VM scheduling communities.

Key Components of VMAgent:

- SchedGym (Simulator): it provides many practical scenarios and flexible configurations to define custom scenarios.
- SchedAgent (Algorithms): it provides many popular RL methods as the baselines.
- SchedVis (Visulization): it provides the visualization of schedlueing dynamics on many metrics.

.. toctree::
    :maxdepth: 1
    :hidden: 
    :caption: Installation

    installation/installation.rst
    
.. toctree::
    :maxdepth: 2
    :hidden: 
    :caption: SchedGym
   
    simulator/scenarios.rst
    simulator/dataset.rst
    simulator/gym.rst
    
.. toctree::
    :maxdepth: 2
    :hidden: 
    :caption: SchedAgent

    algorithms/framework.rst
    algorithms/dqn.rst
    algorithms/a2c.rst
    algorithms/ppo.rst
    algorithms/sac.rst
  
.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: SchedVis
    

    visualization/usage.rst 
    visualization/format.rst 

.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: API Documents

    apidoc/schedgym/modules.rst
    apidoc/vmagent/modules.rst
