from typing import List
import pandas as pd
from umapflow.visualization.PlotColorHelper import ColorManager
from umapflow.umap.umap_model import UMAPModel


class UMAPPlotData:

    def __init__(self, umap: UMAPModel,  plot_data: pd.DataFrame, labels: pd.Series, events: pd.DataFrame = None) -> None:
        self.umap = umap
        self.plot_data = plot_data
        self.labels = labels
        self.events = events

    def setDefaultColoring(self, label_names: List[str]):

        self.stringLabels = ColorManager.create_label_string(self.labels, label_names)
        used_labels = list(set(self.stringLabels))
        self.palette = ColorManager.get_colors_of_default_labels(used_labels)
        self.hue_order = used_labels

    
