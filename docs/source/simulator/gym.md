# GYM
Here we define the state, action, reward function and transition function in the VM scheduling.

## State
SchedAgent make scheduling based on the cluster status and the current request information.
For a cluster has N servers and each server has two numas. 
Each numa has two type of resources: cpu and memory.
The cluster status is represented as a vector with Nx2x2 shape.
For the request, each request contains information about cpu and memory, which makes us represent it with a vector with shape 2.
Thus the whole state represented as [cluster, request] which has (Nx2x2+2) shape.

## Action 
In our VM scheduling, the scheduler is to select which server to handle the current request.
Thus the action space of the scheduler is N.
It should be noted that, if the selected server is unable to handle the request, it will be treated as invalid one.
Due to the double numa architecture, for the small requet (requested cpu is smaller than a threshold) it needs to be allocated on a specific numa of a server. 
This makes the action space to N.
For simplicity, our SchedGym makes the action space 2N and the large request is handled by action%2.

## Reward 
In our VMScheduling, the scheduler is to avoid termination.
We denote a scheduler perform better than others if it can handle more request with the same number of resources.
Thus one of the simplest way is designing reward as '+1' after the scheduler handle a request.

The '+1' reward makes a difficulty on understand the impact of large request.
Thus we propose another reward `request['cpu]`.
The scheduler gain more reward if it handle a larger request.

## Transition Function
When a server handle a creation request (c0, m0), it will allocate (c0, mo) resource for the request.
Specifically, if the server is [[c1,m1], [c2,m2]] and (c0, m0) is a large creation request.
The server will be [[c1-c0/2, m1-m0/2],[c2-c0/2, m2-m0/2]].
If (c0, m0) is a small request and server's first numa is to handle it, then the server will be [[c1-c0, m1-m0],[c2, m2]].
For the deletion request, the minus above will turn to `+`.

