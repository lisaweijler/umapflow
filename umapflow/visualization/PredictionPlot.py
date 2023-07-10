from collections import Counter
from umapflow.utils.plothelper import getOuterAxislimit

from umapflow.visualization.BasePlot import BasePlot
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd
import numpy as np


class PredictionPlot(BasePlot):

    def __init__(self, filepath: str, 
                       caption: str, 
                       labels: np.ndarray, 
                       data_to_plot: pd.DataFrame, 
                       ax: matplotlib.axes = plt.gca(), 
                       use_new_ax=True,
                       overwrite=False):
                       
        super(PredictionPlot, self).__init__(filepath, caption, ax)

        self.labels = labels.astype(int)
        self.data_to_plot = data_to_plot.copy()
        self.use_new_ax = use_new_ax
        self.overwrite = overwrite # render plot again even if it exists if True
        self.palette = sns.color_palette('deep', np.unique(self.labels).max()+1)
        #self.colors = [self.palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in self.cluster_labels] # grey for -1 cluster = unassigned
        self.colors = labels
        # eventsInfoString = "".join(["{}:{} events \n".format(*i) for i in Counter(self.colors).items()])
        
        # self.sublabel = "HDBSCAN \n min_cluster_size: {} \n min_samples : {} \n {}".format(
        #     min_cluster_size, min_samples, eventsInfoString)
        

    def createPlot(self):

        # todo ensure ax scale is same as .umap.embedding_


        if self.use_new_ax:
            fig, ax = plt.subplots(figsize=(30, 30))
        else:
            ax = self.ax

        # if not self.point_types is None:
        #     style_order = [1, 0]
        # else:
        #     style_order = None

        size = 0.4
        
        self.data_to_plot['color'] = self.colors

        g = sns.scatterplot(data=self.data_to_plot, x=0, y=1, hue="color", palette=self.palette,
                            size=size, ax=ax, alpha=0.8)

        embedding = self.data_to_plot.drop(columns='color').to_numpy()
        embeddingX = embedding[:, 0]
        embeddingY = embedding[:, 1]
        x_min, x_max = getOuterAxislimit(g.get_xlim(), (embeddingX.min(), embeddingX.max()))
        y_min, y_max = getOuterAxislimit(g.get_ylim(), (embeddingY.min(), embeddingY.max()))
        g.set_xlim(x_min, x_max)
        g.set_ylim(y_min, y_max)

        #ax.text(0.05, 0.70,  self.sublabel, fontsize=18, transform=ax.transAxes)
        plt.title(self.caption)
        # plt.tight_layout()

        
        

