import numpy as np
import csv

# 处理clean_result到对应的格式
# write_path = 'clean.csv'
# csv_write = csv.writer(open(write_path,'a'))
# column_name = ['vmid','cpu','mem','time','type']
# csv_write.writerow(column_name)

# read_path = 'clean_result.csv'
# f = csv.reader(open(read_path,'r'))
# k=0
# for item in f:
#     if k ==0:
#         k+=1
#         continue
#     row = [item[0],item[1],item[2],item[3],1-int(item[6])]
#     csv_write.writerow(row)


# 处理dataset，让其中的cpu按超分完成1/3
# write_path = 'dataset_deal.csv'
# csv_write = csv.writer(open(write_path,'a'))

# column_name = ['vmid','cpu','mem','time','type']
# csv_write.writerow(column_name)

# read_path = 'dataset.csv'
# f = csv.reader(open(read_path,'r'))
# k=0
# for item in f:
#     if k ==0:
#         k+=1
#         continue
#     c = int(item[1])
#     m = int(item[2])

#     prob = 1.1
#     if [c,m] in [[1,2],[4,16],[1,1],[2,8],[8,32],[1,4]]:
#         prob = 0.9
#     if c==2 and m==4:
#         prob = 0.65
#     if c==4 and m==8:
#         prob = 0.75
#     if c==8 and m==16:
#         prob = 0.6    
#     a = np.random.uniform()
#     if a < prob:
#         c = round(c/3,1)
    
#     row = [item[0],c,item[2],item[3],item[4]]
#     csv_write.writerow(row)


write_path = 'dataset_trible.csv'
csv_write = csv.writer(open(write_path,'a'))

column_name = ['vmid','cpu','mem','time','type']
csv_write.writerow(column_name)

read_path = 'dataset_deal.csv'
f = csv.reader(open(read_path,'r'))
k=0
for item in f:
    if k ==0:
        k+=1
        continue
    c = int(item[1])
    c = round(c/3,1)
    
    row = [item[0],c,item[2],item[3],item[4]]
    csv_write.writerow(row)