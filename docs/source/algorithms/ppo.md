# PPO

## conda env
* `conda activate VMAgent`

## Example
Train PPO in fading environment with 5 servers, and parameters gamma=0.99 learning_rate=0.003:
```
python vmagent/train.py --env fading --alg ppo --N 5 --gamma 0.99 --lr 0.003
```

Train PPO in recovering environment with 5 servers, and parameters gamma=0.99 learning_rate=0.003:
```
python vmagent/train.py --env recovering --alg ppo  --N 5 --gamma 0.99 --lr 0.003
```

## Visualization

For visualization, see the [`render`](./render) directory in detail.
