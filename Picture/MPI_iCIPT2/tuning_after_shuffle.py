#!/usr/bin/env python3

import matplotlib.pyplot as plt

# x = [1, 4, 8, 16, 32]
# y = [95.2965, 92.4334, 96.5011, 96.2478, 96.9494]

x = [1, 8, 16, 32]
x = [x0 * 16 for x0 in x]
y = [95.2965, 96.5011, 96.2478, 96.9494]

plt.figure(figsize=(8, 7), dpi=100)
plt.xticks([1*16, 4*16, 8*16, 16*16, 32*16],fontsize=15)
#plt.ylim(80, 100)
plt.yticks([95, 95.5, 96, 96.5, 97, 97.5],fontsize=15)
# plt.yticks()
plt.plot(x, y, marker='o', markersize=12,linewidth=3)
#plt.title("MPI ENPT2 Tuning(after shuffle)", fontsize=15)      
plt.xlabel("Number of Tasks", fontsize=20)   
plt.ylabel("Parallel Efficiency(%)", fontsize=20)       
# plt.yrange([94.5,97.5])

plt.savefig("tuning_after_shuffle.png", dpi=600, bbox_inches='tight')
