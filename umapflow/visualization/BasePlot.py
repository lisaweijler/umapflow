import abc
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path


class BasePlot(abc.ABC):
    def __init__(self, filepath: str, caption: str, ax: matplotlib.axes = plt.gca()):
        self.filepath = filepath
        self.caption = caption
        font = {
            # 'weight' : 'bold',
            'size': 14}
        self.ax = ax
        matplotlib.rc('font', **font)

    def showPlot(self):
        plt.show()
        plt.clf()
        plt.cla()
        plt.close()

    def savePlot(self):
        plt.savefig(self.filepath)
        plt.cla()
        plt.clf()
        plt.close()
        print(f"INFO: successfully saved plot with caption {self.caption} to path: {self.filepath}")

    @abc.abstractmethod
    def createPlot(self):
        pass

    def generatePlotFile(self):
        # check if plot already exists -> dont render again
        if Path(self.filepath).exists() and not self.overwrite:
            print(f"Plot {self.filepath} already exists -> skipped")
            return
        self.createPlot()
        self.savePlot()
