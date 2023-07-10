import pandas as pd
from umapflow import DataType


class ClusterInfo:
    """
    Holds useful information about a HDBSCAN cluster
    such as: n_blasts, n_non-blasts, n_events, cluster center, etc.
    """

    def __init__(self) -> None:
        pass

    def generate_info_from_cluster_data(self, cluster_label: int, 
                                              events_in_cluster: pd.DataFrame, 
                                              binary_labels: pd.Series, 
                                              data_type: pd.Series):

        self.cluster_label = cluster_label
        self.n_events = len(events_in_cluster)

        if self.n_events != len(binary_labels):
            raise ValueError("labels and events must have the same length!!")

        self.n_mrd = (data_type == DataType.TEST.value).sum()
        self.n_transform = (data_type == DataType.TRANSFORM.value).sum()
        self.n_control = (data_type == DataType.CONTROL.value).sum()
        self.n_control_transform = self.n_events - self.n_mrd

        self.ratio = self.n_mrd/(self.n_control_transform + self.n_mrd)

        self.n_blasts = (binary_labels == 1).sum()
        self.n_non_blasts = self.n_events - self.n_blasts

        cluster_median = events_in_cluster.median()

        self.cluster_center = (cluster_median[0], cluster_median[1])
