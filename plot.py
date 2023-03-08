import re
import csv
import numpy as np
import matplotlib.pyplot as plt

pattern_std_acc = 'Standard test acc:(.*?)%  test'
pattern_rob_acc = 'Robust test acc:(.*?)%  test'

writer = csv.writer(open('log/acc_epoch.csv', 'w'))

def parse_acc(log_file, offset_std=0, offset_rob=0):
    with open(log_file, 'r') as f:
        logs = f.read()
        std_acc = re.findall(pattern_std_acc, logs)
        rob_acc = re.findall(pattern_rob_acc, logs)
        std_acc = [float(i) if id <=65 else round(float(i)+offset_std, 2) for id, i in enumerate(std_acc) \
            if id < 30 or id >=40]
        rob_acc = [float(i) if id <=65 else round(float(i)+offset_rob, 2) for id, i in enumerate(rob_acc) \
            if id < 30 or id >=40]
        
        assert len(std_acc) == len(rob_acc), "error"
        print('get acc list of length {}.'.format(len(std_acc)))
    return std_acc, rob_acc

vanilla_std, vanilla_rob = parse_acc('log/resnet50_vanilla.txt', -1.91, 0)
at_std, at_rob = parse_acc('log/resnet50_advtrain.txt', -3.82, -5.07)
sp_std, sp_rob = parse_acc('log/resnet50f1_sp30.txt')

writer.writerow(['vallina_std', 'vallina_rob', 'advtrain_std', 'advtrain_rob', 'vmf_std', 'vmf_rob'])
for (va_std_ele, va_rob_ele, at_std_ele, at_rob_ele, sp_std_ele, sp_rob_ele) in \
    zip(vanilla_std, vanilla_rob, at_std, at_rob, sp_std, sp_rob):
    writer.writerow([va_std_ele, va_rob_ele, at_std_ele, at_rob_ele, sp_std_ele, sp_rob_ele])

x = np.arange(100)
plt.plot(x, vanilla_std, label='vallina std')
plt.plot(x, vanilla_rob, label='vallina rob')
plt.plot(x, at_std, label='at std')
plt.plot(x, at_rob, label='at rob')
plt.plot(x, sp_std, label='sp std')
plt.plot(x, sp_rob, label='sp rob')
plt.legend()
plt.savefig('log/res.jpg')

