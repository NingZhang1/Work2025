#!/usr/bin/env python3

import matplotlib.pyplot as plt

# x = [1, 4, 8, 16, 32]
# y = [45.9360, 88.1810, 94.0272, 93.6192, 93.5170]

x = [1, 8, 16, 32]
x = [x0 * 16 for x0 in x]
y = [45.9360, 94.0272, 93.6192, 93.5170]

plt.figure(figsize=(8, 7), dpi=100)
plt.xticks([1*16, 4*16, 8*16, 16*16, 32*16],fontsize=15)
#plt.ylim(80, 100)
#plt.yticks([80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100])
plt.yticks(fontsize=15)
plt.plot(x, y, marker='o', markersize=12,linewidth=3)
#plt.title("MPI ENPT2 Tuning(after shuffle)", fontsize=15)      
plt.xlabel("Number of Tasks", fontsize=20)    
plt.ylabel("Parallel Efficiency(%)", fontsize=20)       

plt.savefig("tuning_before_shuffle.png", dpi=600, bbox_inches='tight')
