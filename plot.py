import matplotlib.pyplot as plt
import numpy as np
lines = np.loadtxt("episode_reward_3.txt", comments="#", delimiter="\n", unpack=False)
lines_2 = np.loadtxt("episode_reward_2.txt", comments="#", delimiter="\n", unpack=False)

plt.plot(lines)
plt.plot(lines_2)
plt.show()
