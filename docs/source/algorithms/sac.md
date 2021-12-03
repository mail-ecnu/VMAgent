# SAC

## conda env
* `conda activate VMAgent`

## Example
Train SAC in fading environment with 5 servers, and parameters gamma=0.99 learning_rate=0.003:
```
python vmagent/train.py --env fading --alg sac --N 5 --gamma 0.99 --lr 0.003
```

Train SAC in recovering environment with 5 servers, and parameters gamma=0.99 learning_rate=0.003:
```
python vmagent/train.py --env recovering --alg sac  --N 5 --gamma 0.99 --lr 0.003
```

## Visualization

For visualization, see the [`render`](./render) directory in detail.
