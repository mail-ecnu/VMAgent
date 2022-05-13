import numpy as np
import csv


# 处理dataset，让其中的cpu完成1/3
write_path = 'dataset_deal2.csv'
csv_write = csv.writer(open(write_path,'a'))

column_name = ['vmid','cpu','mem','time','type']
csv_write.writerow(column_name)

read_path = 'Huawei-East-1.csv'
f = csv.reader(open(read_path,'r'))
k=0

chaofen_id = []
for item in f:
    if k ==0:
        k+=1
        continue
    c = int(item[1])
    m = int(item[2])

    prob = -1
    if [c,m] in [[1,2],[4,16],[1,1],[2,8],[8,32],[1,4]]:
        prob = 0.9
    if c==2 and m==4:
        prob = 0.65
    if c==4 and m==8:
        prob = 0.75
    if c==8 and m==16:
        prob = 0.6    
    a = np.random.uniform()

    # 过滤掉1U的小请求
    # if c == 1:
    #     continue
    if a < prob or item[0] in chaofen_id:
        c = round(c/3,1)
        chaofen_id.append(item[0])
    
    row = [item[0],c,item[2],item[3],item[4]]
    

    csv_write.writerow(row)