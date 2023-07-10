from umapflow.clustering.cluster_info import ClusterInfo
from umapflow.utils.plothelper import getOuterAxislimit
from umapflow.visualization.BasePlot import BasePlot

from collections import Counter
from typing import List
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd
import numpy as np


class HDBSCANPlot(BasePlot):

    def __init__(self, filepath: str,
                 caption: str,
                 cluster_labels: np.ndarray,
                 data_to_plot: pd.DataFrame,
                 min_cluster_size: int,
                 min_samples=None,  # can be non
                 cluster_info: List[ClusterInfo] = [],
                 ax: matplotlib.axes = plt.gca(),
                 use_new_ax=True,
                 overwrite=False):

        super(HDBSCANPlot, self).__init__(filepath, caption, ax)

        self.cluster_labels = cluster_labels
        self.data_to_plot = data_to_plot.copy()
        self.cluster_info = cluster_info
        self.use_new_ax = use_new_ax
        self.overwrite = overwrite  # render plot again even if it exists if True
        self.palette = sns.color_palette('deep', len(np.unique(self.cluster_labels)))
        # self.colors = [self.palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in self.cluster_labels] # grey for -1 cluster = unassigned
        self.colors = cluster_labels
        eventsInfoString = "".join(["{}:{} events \n".format(*i) for i in Counter(self.colors).items()])

        self.sublabel = "HDBSCAN \n min_cluster_size: {} \n min_samples : {} \n {}".format(
            min_cluster_size, min_samples, eventsInfoString)

    def add_cluster_info_labels(self, ax: matplotlib.axes):

        for cluster_info in self.cluster_info:
            if cluster_info.n_blasts > 0:
                label_text = f"n_events {cluster_info.n_events} n_blasts {cluster_info.n_blasts}\n"
                label_text += f"n_control_transform {cluster_info.n_control_transform} n_mrd  {cluster_info.n_mrd}"
                ax.text(cluster_info.cluster_center[0], cluster_info.cluster_center[1], label_text, fontsize=14, transform=ax.transData)

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
        self.add_cluster_info_labels(ax)
        plt.title(self.caption)
        # plt.tight_layout()
