import time
import psutil
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

class RealTimeMonitoring:
    def __init__(self):
        self.cpu_usage = []
        self.mem_usage = []
        self.disk_usage = []
        self.net_io = []
        self.fig, self.ax = plt.subplots(4, 1, figsize=(10, 10))

    def monitor(self):
        while True:
            self.cpu_usage.append(psutil.cpu_percent())
            self.mem_usage.append(psutil.virtual_memory().percent)
            self.disk_usage.append(psutil.disk_usage('/').percent)
            self.net_io.append(psutil.net_io_counters().bytes_sent + psutil.net_io_counters().bytes_recv)

            self.ax[0].cla()
            self.ax[0].plot(self.cpu_usage)
            self.ax[0].set_title('CPU Usage')
            self.ax[0].set_ylim(0, 100)

            self.ax[1].cla()
            self.ax[1].plot(self.mem_usage)
            self.ax[1].set_title('Memory Usage')
            self.ax[1].set_ylim(0, 100)

            self.ax[2].cla()
            self.ax[2].plot(self.disk_usage)
            self.ax[2].set_title('Disk Usage')
            self.ax[2].set_ylim(0, 100)

            self.ax[3].cla()
            self.ax[3].plot(self.net_io)
            self.ax[3].set_title('Network I/O')
            self.ax[3].set_ylim(0, 1000000)

            plt.pause(1)

    def run(self):
        self.monitor()

if __name__ == '__main__':
    monitor = RealTimeMonitoring()
    monitor.run()
