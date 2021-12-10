# Dataset
Our VMAgent is constructed based on one month real VM scheduling dataset called [*Huawei-East-1*](https://vmagent.readthedocs.io/en/latest/simulator/dataset.html) from [**HUAWEI Cloud**](https://www.huaweicloud.com).
The data format is concluded below

| Field  | Type | Description                                  |
|--------|------|----------------------------------------------|
| `vmid`   | int  | The virtual machine ID                       |
| `cpu`    | int  | Number of CPU cores                          |
| `memory` | int  | Number of Memory GBs                         |
| `time`   | int  | Relative time in seconds                     |
| `type`   | int  | 0 denotes creation while 1 denotes deleteion |

For more information about the dataset, read the [doc](https://vmagent.readthedocs.io/en/latest/simulator/dataset.html#data-format)
