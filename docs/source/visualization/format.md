# Data Formats

## Supported Data Formats

We support data files in two formats: `Data` or `Raw Data` in a single pickle file.

## Objects

The structure and necessary fileds of each object are described below.

### Data

| Field | Type | Description |
| ---- | ---- | ---- |
| `name` | `string` | Scheduling algorithm name |
| `data` | `List[Frame]` (a.k.a `Raw Data`) |   |

### Raw Data

```python
[
    Frame1,
    Frame2,
    Frame3,
    ...
]
```

### Frame

| Field | Type | Description |
| ---- | ---- | ---- |
| `server` | `List[Server]` | The status of each server at the current time. |
| `request` | `Request` |  The info of the request at the current time. |
| `action` | `int` | The resource id to which the current request is assigned, <br> which is calculated by `server id * 2 + numa id`. |

### Server

```python
[
    [CPU1, MEM1], # CPU and MEM usage of NUMA1
    [CPU2, MEM2]  # CPU and MEM usage of NUMA2
]
```

### Request

| Field | Type | Description |
| ---- | ---- | ---- |
| `cpu` | `int` |  Required CPU  |
| `mem` | `int` |  Required memory  |
| `type` | `int` | `0` for `allocation` and `1` for `release` |  
| `is_double` | `bool` or `int` | Whether the request is a double-numa request or not |

