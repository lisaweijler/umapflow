from collections import Counter
from umapflow.visualization.umap_plotdata import UMAPPlotData
from umapflow.utils.plothelper import getOuterAxislimit
from umapflow.visualization.BasePlot import BasePlot
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd


class UMAPPlot(BasePlot):

    def __init__(self, filepath: str, 
                       caption: str,
                       umap_data: UMAPPlotData,
                       point_types=None, 
                       ax: matplotlib.axes = plt.gca(), 
                       use_new_ax=True,
                       overwrite=False):
                       
        super(UMAPPlot, self).__init__(filepath, caption, ax)

        self.umap = umap_data.umap
        self.data_to_plot = umap_data.plot_data.copy()
        self.data_to_plot["color"] = umap_data.stringLabels
        self.palette = umap_data.palette
        self.hue_order = umap_data.hue_order
        self.use_new_ax = use_new_ax
        self.point_types = point_types
        self.overwrite = overwrite # render plot again even if it exists if True
        eventsInfoString = "".join(["{}:{} events \n".format(*i) for i in Counter(umap_data.stringLabels).items()])
        umapTypeString = "parametric UMAP" if hasattr(self.umap, "use_parametric_umap") and self.umap.use_parametric_umap else "UMAP"
        self.sublabel = "{} \n n_neighbors: {} \n min_dist : {} \n op_mix_ratio: {} \n {}".format(
            umapTypeString, self.umap.n_neighbors, self.umap.min_dist, self.umap.set_op_mix_ratio, eventsInfoString)
        

    def createPlot(self):

        # todo ensure ax scale is same as .umap.embedding_

        size = 1.2

        if "size" in self.data_to_plot.columns:
            size = [0.6 if size == 3 else 0.4 for size in self.data_to_plot["size"]]

        if self.use_new_ax:
            fig, ax = plt.subplots(figsize=(30, 30))
        else:
            ax = self.ax

        # if not self.point_types is None:
        #     style_order = [1, 0]
        # else:
        #     style_order = None

        g = sns.scatterplot(data=self.data_to_plot, x=0, y=1, hue="color", palette=self.palette, hue_order=self.hue_order, style=self.point_types,
                            size=size, ax=ax, alpha=0.8)

        embeddingX = self.umap.umap.embedding_[:, 0]
        embeddingY = self.umap.umap.embedding_[:, 1]
        x_min, x_max = getOuterAxislimit(g.get_xlim(), (embeddingX.min(), embeddingX.max()))
        y_min, y_max = getOuterAxislimit(g.get_ylim(), (embeddingY.min(), embeddingY.max()))
        g.set_xlim(x_min, x_max)
        g.set_ylim(y_min, y_max)

        ax.text(0.05, 0.70,  self.sublabel, fontsize=18, transform=ax.transAxes)
        plt.title(self.caption)
        # plt.tight_layout()
