import numpy as np
import matplotlib.pyplot as plt
import json

with open('result/bf.json', 'r') as f:
    bf = json.load(f)

with open('result/bb.json', 'r') as f:
    bb = json.load(f)

x = list(range(3, 14))

p_bf = np.polyfit(x[:len(bf)], bf, len(bf))
bf = bf + [np.polyval(p_bf, 12), np.polyval(p_bf, 13)]

plt.semilogy(x, bf, 'b', label='brute-force')
plt.semilogy(x, bb, 'r', label='branch&bound')

plt.title('Branch&Bound vs Brute-Force')
plt.ylabel('secondi')
plt.xlabel('nodi')
plt.legend()
plt.grid()

plt.savefig('result/versus.png', dpi=1200)

plt.savefig()
