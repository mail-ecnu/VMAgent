# Installation
## Install from Source
First clone our git repo: 

```sh
git clone git@github.com:mail-ecnu/VMAgent.git
cd VMAgent
```

Then create the virtual environment to satisfy dependency with conda:

```sh 
conda env create -f conda_env.yml
conda activate VMAgent-dev
```
Finally Install our simulator:

```sh
python3 setup.py develop
````