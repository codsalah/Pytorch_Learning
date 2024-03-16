import torch
import matplotlib.pyplot as plt
from scipy.stats import binom

p = 0.2
ns = [1, 10, 100, 1000]
plt.figure(figsize=(10, 3))

for i in range(4):
    n = ns[i]
    pmf = torch.tensor([p**i * (1-p)**(n-i) * binom.pmf(i, n, p)
                        for i in range(n + 1)])
    plt.subplot(1, 4, i + 1)
    plt.stem([(i - n*p)/torch.sqrt(torch.tensor(n*p*(1 - p)))
              for i in range(n + 1)], pmf.numpy(),
             use_line_collection=True)
    plt.xlim([-4, 4])
    plt.xlabel('x')
    plt.ylabel('p.m.f.')
    plt.title("n = {}".format(n))

plt.show()