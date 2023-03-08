import csv 
import numpy as np
import matplotlib.pyplot as plt

reader = csv.reader(open('acc_epoch.csv', 'r'))
next(reader)

data = []
for line in reader:
    line = [float(i) for i in line]
    data.append(line)

vanilla_std, vanilla_rob, at_std, at_rob, sp_std, sp_rob = zip(*data)

x = np.arange(100)
plt.plot(x, vanilla_std, label='vallina std')
plt.plot(x, vanilla_rob, label='vallina rob')
plt.plot(x, at_std, label='at std')
plt.plot(x, at_rob, label='at rob')
plt.plot(x, sp_std, label='sp std')
plt.plot(x, sp_rob, label='sp rob')
plt.legend()
plt.savefig('res.jpg')


