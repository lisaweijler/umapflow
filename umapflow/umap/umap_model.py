from umapflow.utils.picklefilehandlers import PickleFileLoader, PickleFileWriter
from umapflow.utils.executiontimer import executiontimer
from umapflow.utils.util import ptictoc

import numpy as np
import umap
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path

class UMAPModel:

    @staticmethod
    def load(filePath: Path):
        if filePath.is_file():  # file --> normal umap
            print("INFO: loading normal umap from file: {}".format(filePath))
            fileLoader = PickleFileLoader(filePath)
            return fileLoader.loadPickleFromInputfile()
        else:
            raise FileNotFoundError(f"No umap model found: {filePath}")

    def __init__(self, name: Path, basepath: Path, control_test_data_labels: pd.DataFrame, n_components=2, n_neighbors=10, min_dist=0.1, op_mix_ratio=1.0, metric='euclidean', semi_supervised=False, use_parametric_umap=False):
        self.name = name
        self.basepath = basepath
        self.control_test_data_labels = control_test_data_labels # stored here in case it is loaded from pickle, so we are sure to use the same order of events!
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.set_op_mix_ratio = op_mix_ratio
        self.metric = metric
        self.semi_supervised = semi_supervised
        self.use_parametric_umap = use_parametric_umap
        self._embedding = None
        self._transformed_data = None

    @executiontimer
    def fitUmap(self, x_data: np.ndarray, target: np.ndarray = None):

        if not hasattr(self, "umap"):
            self.umap = umap.UMAP(n_components=self.n_components, 
                                  n_neighbors=self.n_neighbors, 
                                  min_dist=self.min_dist, 
                                  set_op_mix_ratio=self.set_op_mix_ratio, 
                                  random_state=42,
                                  metric=self.metric)

        if not hasattr(self, "scaler"):
            # self.scaler = MinMaxScaler()
            self.scaler = StandardScaler()

        x_data_scaled = self.scaler.fit_transform(x_data)
        #x_data_scaled = x_data
        ptictoc()
        if self.semi_supervised:
            self.umap.fit(x_data_scaled, y=target)
        else:
            self.umap.fit(x_data_scaled)
        ptictoc('umap-fitting')
        self._embedding = self.umap.embedding_
        

    @executiontimer
    def projectData(self, data):

        if not hasattr(self, "umap") or self.umap == None:
            raise Exception("umap object not initialized")

        if not hasattr(self, "scaler"):
            raise Exception("scaler object not initialized")

        scaled_data = self.scaler.transform(data)
        #scaled_data = data
        ptictoc()
        self._transformed_data = self.umap.transform(scaled_data)
        ptictoc('umap-transform')
        

    def save(self):

        fileName = self.name + "-umap.pkl"
        fileWriter = PickleFileWriter(os.path.join(self.basepath, fileName))
        fileWriter.writePickleToOutputfile(self)

    def get_embedding(self):
        if self._embedding is None:
            raise Exception("No embedding yet.. call fitUmap() first!")
        return self._embedding

    def get_transformed_data(self):
        if self._transformed_data is None:
            raise Exception("No additional data transformed yet.. call projectData() first!")
        return self._transformed_data

    
