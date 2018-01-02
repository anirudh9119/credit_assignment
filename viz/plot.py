import matplotlib.pyplot as plt
import numpy as np
import glob
qw = glob.glob('*.out')


color_defaults = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf',  # blue-teal
    '#bcbd22',  # curry yellow-green
    '#17becf'  # blue-teal
]



for i in range(len(qw)):
    open('file_' + np.str(i) + '.txt','w').writelines([ line for line in open(qw[i]) if 'episode' in line])


for i in range(len(qw)):
    print i
    lines = np.loadtxt('file_' + np.str(i) + '.txt', delimiter=" ", usecols=(4), unpack=True)
    plt.plot(lines, color=color_defaults[i], label=qw[i])

plt.legend(loc='best')
plt.show()
