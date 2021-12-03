# A2C

## conda env
* `conda activate VMAgent`

## Example
Train A2C in fading environment with 5 servers, and parameters gamma=0.99 learning_rate=0.003:
```
python vmagent/train.py --env fading --alg a2c --N 5 --gamma 0.99 --lr 0.003
```

Train A2C in recovering environment with 5 servers, and parameters gamma=0.99 learning_rate=0.003:
```
python vmagent/train.py --env recovering --alg a2c  --N 5 --gamma 0.99 --lr 0.003
```

## Visualization

For visualization, see the [`render`](./render) directory in detail.
