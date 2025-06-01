#!/usr/bin/env python3

import matplotlib.pyplot as plt

M = 100 * 10000

x = [24 * M, 53 * M, 102 * M, 309 * M, 539 * M] 
x = [y/ (100 * M) for y in x]
y = [-2099.863816, -2099.875327, -2099.883027, -2099.893761, -2099.898165]

x1 = [15384848,25268761,53841876,272704992]
x1 = [x2/ (100 * M) for x2 in x1]
y1 = [-2099.86657636,-2099.87331369,-2099.88232915,-2099.89719109]


plt.figure(figsize=(16, 9), dpi=100)
# plt.xticks([40 * M, 80 * M, 160 * M, 273 * M, 320 * M, 539 * M], fontsize=15)
plt.xticks([0.4, 0.8, 1.6, 2.73, 3.2, 5.39], fontsize=20)
# plt.ylim(80, 100)
plt.yticks([-2099.865, -2099.870, -2099.875, -2099.880, -2099.885, -2099.890, -2099.895], fontsize=20)
plt.plot(x, y, marker='o', label="SHCI", markersize=15, linewidth=3)
plt.plot(x1, y1, marker='o', label="iCIPT2", markersize=15, linewidth=3)
#plt.title("MPI ENPT2 Tuning(after shuffle)", fontsize=15)      
plt.xlabel("Size of Variational Space (in $10^8$)", fontsize=25)    
plt.ylabel("$E_{\\text{var}}/E_h$", fontsize=25)       
plt.legend(fontsize=25)
# plt.xscale('log')
plt.savefig("tuning_after_shuffle.png", dpi=600, bbox_inches='tight')
