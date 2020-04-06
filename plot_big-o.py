from scipy.special import gamma
from matplotlib import pyplot as plt
import numpy as np
import math

common = dict(linewidth=2)

n = np.linspace(1,10,100)
plt.plot(n, gamma(n),
    color='darkred',
    label='O(n!)',
    **common
    )

n = np.linspace(1,20,100)
plt.plot(n, 2**n,
    color='tomato',
    label=r'O($2^{\mathrm{n}}$)',
    **common
    )

n = np.linspace(1,100,100)
plt.plot(n, n**2,
    color='orange',
    label=r'O(n$^2$)',
    **common
    )

n = np.linspace(1,300,100)
plt.plot(n, n*np.log2(n),
    color='gold',
    label='O(n log(n))',
    **common
    )

plt.plot(n, n,
    color='yellow',
    label='O(n)',
    **common
    )

plt.plot(n, np.log2(n),
    color='greenyellow',
    label='O(log(n))',
    **common
    )

plt.plot(n, [1]*len(n),
    color='limegreen',
    label='O(1)',
    **common
    )


plt.tick_params(
    axis='both',
    which='both',
    left=False,
    bottom=False,
    top=False,
    labelleft=False,
    labelbottom=False
    )

plt.ylim(-100,10000)
plt.xlim(5,300)
plt.xlabel('number of elements, n')
plt.ylabel('number of operations')
plt.grid()
plt.legend()

plt.savefig('big-o.pdf')