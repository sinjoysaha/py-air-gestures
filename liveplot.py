import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# fieldnames = ["timestamp", "index_tip_x", "index_tip_y"]

def animate(frame):
    mhl_data = pd.read_csv('mhl_data.csv')

    plt.cla()
    plt.plot(mhl_data["timestamp"], mhl_data["index_tip_x"], label="Index Tip X")
    plt.plot(mhl_data["timestamp"], mhl_data["index_tip_y"], label="Index Tip Y")

    plt.legend(loc='upper left')
    plt.tight_layout()

ani = FuncAnimation(plt.gcf(), animate, interval=1000)
# plt.tight_layout()
plt.show()