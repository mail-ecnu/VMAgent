import csv
import numpy as np

f = open('search.csv','w')
writer = csv.writer(f)
step_list = []

for i in range(10000):
    step_list.append(np.random.randint(0,200000))

# for i in range(50):
#     step_list.append(np.random.randint(0,200000))

# f = csv.reader(open('ok_step.csv','r'))
# print(f)
# train_steps = []
# for item in f:
#     train_steps = item
# for i in range(len(train_steps)):
#     train_steps[i] = int(train_steps[i])
# for i in train_steps:
#     if i<0 or i>200000:
#         print(i)
# print(len(train_steps))

writer.writerow(step_list)


