# [![VMAgent LOGO](./docs/source/images/logo.svg)](https://VNAgent.readthedocs.io/en/latest/)

VMAgent is a platform for exploiting Reinforcement Learning (RL) on Virtual Machine (VM)Scheduling tasks.
It contains multiple practicle VM scheduling scenarios (such as Fading, Rcovering, etc).
These scenarios also correspond to the challanges in the RL.
Exploiting the design of RL methods in these secenarios help both the RL and VM scheduling community.

Key Components of VMAgent:
* Simulator: it provides many practical scenarios and flexible configurations to define custom scenarios.
* RL Library: it provides many popular RL methods as the baselines.
* Dashboard: it provides the visualization of schedlueing dynamics on many metrics.
## Scenarios and Baselines

The VMAgent provides multiple practical scenarios: 
| Scenario     | Allow-Deletion | Allow-Expansion | Allow-MultiNuma | Server Num |
|--------------|----------------|-----------------|-----------------|------------|
| Fading       | False          | False           | False           | Small      |
| Recovering   | True           | False           | False           | Small      |
| Expanding    | True           | True            | False           | Small      |
| Recovering-L | True           | False           | False           | Large      |
| Fading-N     | False          | False           | True            | Small      |

Researchers can also flexibly customized their scenarios in the `vmagent/config/` folder.


Besides, we provides many baselines for quick startups.
It includes FirstFit, BestFit, DQN, PPO, A2C and SAC.
More baselines is coming.
## Installation 

### Install from PyPI

TBA

### Install from Source

```
git clone git@github.com:mail-ecnu/VMAgent.git
cd VMAgent
conda env create -f conda_env.yml
conda activate VMAgent-dev
python3 setup.py develop
```

## Quick Examples

In this quick example, we show how to train a dqn agent in a fading scenario. 
For more examples and the configurations' concrete definitions, we refer readers to our [docs](https://VNAgent.readthedocs.io/en/latest/).

config/fading.yaml:
```yaml
N: 5
cpu: 40 
mem: 90
allow_release: False
```
config/algs/dqn.yaml:
```yaml
mac: 'vectormac'
learner: 'q_learner'
agent: 'DQNAgent'
```
Then 
```sh
python train.py --env=fading --alg=dqn
```

It provides the first VM scheudling simulator based on the one month east china data in huawei cloud.
It includes three scenarios in practical cloud: Recovering, Fading and Expansion.
Our video is at [video](https://drive.google.com/file/d/14EkVzUnEXM7b8YNJiZ6cxLxhcj5yW4V_/view?usp=sharing).
Some demonstrations are listed:

<img src="./docs/source/images/rec-small.gif" width="250"><img src="./docs/source/images/rec-large.gif" width="250"><img src="./docs/source/images/exp-large.gif" width="250">
## Installation


## Docs

We present 

## Data 

We collect one month scheduling data in east china region of huawei cloud.
The format and the stastical analysis of the data are presented in the docs.
one month east china data in huawei cloud.

## Visualization

For visualization, see the [`render`](./render) directory in detail.
