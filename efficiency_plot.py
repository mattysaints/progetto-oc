import numpy as np
import matplotlib.pyplot as plt
import json

with open('versus/bf.json', 'r') as f:
    bf = json.load(f)

with open('versus/bb.json', 'r') as f:
    bb = json.load(f)

with open('versus/bb_sparse.json', 'r') as f:
    bb_sparse = json.load(f)

x = list(range(3, 14))

p_bf = np.polyfit(x[:len(bf)], bf, len(bf))
bf = bf + [np.polyval(p_bf, 12), np.polyval(p_bf, 13)]

plt.semilogy(x, bf, 'b', label='brute-force')
plt.semilogy(x, bb, 'r', label='branch&bound')

plt.title('Branch&Bound vs Brute-Force')
plt.ylabel('log(secondi)')
plt.xlabel('nodi')
plt.legend()
plt.grid()

plt.savefig('versus/versus.png', dpi=1200)

plt.semilogy(x, bb_sparse, 'g', label='branch&bound sparse')
plt.legend()

plt.savefig('versus/versus_sparse.png', dpi=1200)
