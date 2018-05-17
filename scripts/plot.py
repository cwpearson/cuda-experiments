#! env python3

import json
from pprint import pprint
import matplotlib.pyplot as plt
import sys
import numpy as np
import pandas as pd

with open(sys.argv[1]) as f:
    df = pd.read_csv(f)

df['bw'] = df['count'] / df['time']

print(df)

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="ticks")

# Initialize the figure with a logarithmic x axis
f, ax = plt.subplots(figsize=(7, 6))
# ax.set_xscale("log")

# Load the example planets dataset
# planets = sns.load_dataset("planets")

# Plot the orbital period with horizontal boxes
sns.boxplot(x="count", y="bw", data=df,
            whis=np.inf, palette="vlag")

# Add in points to show each observation
sns.swarmplot(x="count", y="bw", data=df,
              size=2, color=".3", linewidth=0)

# Tweak the visual presentation
ax.xaxis.grid(True)
ax.set(ylabel="")
sns.despine(trim=True, left=True)

plt.show()
