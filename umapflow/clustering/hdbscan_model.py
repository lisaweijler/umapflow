from umapflow.clustering.cluster_info import ClusterInfo
from umapflow.utils.picklefilehandlers import PickleFileLoader, PickleFileWriter
from umapflow.utils.executiontimer import executiontimer
from umapflow.utils.util import ptictoc

import pandas as pd
import hdbscan
from pathlib import Path
from typing import List
import os





class HDBSCANModel:

    '''
    clusters the umap embedding and holds that information
    is saved/loaded for each sample
    '''

    @staticmethod
    def load(filePath: Path):
        if filePath.is_file():  # file --> normal umap
            print("INFO: loading hdbscan model from file: {}".format(filePath))
            fileLoader = PickleFileLoader(filePath)
            return fileLoader.loadPickleFromInputfile()
        else:
            raise FileNotFoundError(f"No umap model found: {filePath}")

    def __init__(self, name: Path, basepath: Path, 
                       min_cluster_size=20, min_samples=None, transform_strength=None, 
                       data_type='LAIP', overwrite=False) -> None:
        
        self.hdbscan = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, 
                                       min_samples=min_samples,
                                       prediction_data=True) # hdbscan does little extra computation when fitting the model that speeds up prediction(=transfrom) later

        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.transform_strength = transform_strength
        self.data_type = data_type
        self.name = name
        self.overwrite = overwrite
        self.basepath = basepath  # here it gets actual basepath but in future it should only get model base path and plot should be seperatly
        self._predicted_clusters_mrd = None
        self._predicted_clusters_transform_control = None
        self._cluster_info = None

    @executiontimer
    def cluster_data(self, embedded_events: pd.DataFrame) -> None:
        """
        fits the hdbscan clustere with the mrd events
        needs to be called before transform-data
        saves pd.Series with cluster assigments for each event
        """
        ptictoc()
        self.hdbscan.fit(embedded_events)
        ptictoc('hdbscan-fitting')
        self._predicted_clusters_mrd = pd.Series(self.hdbscan.labels_, name="hdbscan_clusters")
    
    @executiontimer
    def transform_data(self, embedded_events: pd.DataFrame) -> None:
        """
        transforms new datapoints into the learnt clustering
        with the fitted HDBSCAN instance
        saves pd.Series with cluster assigments for each event
        """
        ptictoc()
        labels, strength = hdbscan.approximate_predict(self.hdbscan, embedded_events)
        ptictoc('hdbscan-transform')
        if self.transform_strength is not None:
            labels[strength <= self.transform_strength] = -1

        self._predicted_clusters_transform_control = pd.Series(labels, name="hdbscan_clusters")

    def get_predicted_clusters_mrd(self):
        if self._predicted_clusters_mrd is None:
            raise Exception("No clusters predicted yet.. call cluster_data() first!")
        return self._predicted_clusters_mrd

    def get_predicted_clusters_transform_control(self):
        if self._predicted_clusters_transform_control is None:
            raise Exception("No data transfromed yet.. call transform_data() first!")
        return self._predicted_clusters_transform_control

    def get_cluster_info(self):
        if self._cluster_info is None:
            raise Exception("No cluster info generated yet.. call generate_cluster_info() first!")
        return self._cluster_info


    def generate_cluster_info(self, data_mrd: pd.DataFrame,
                                    data_transform_control: pd.DataFrame,
                                    binary_labels_mrd: pd.Series, 
                                    binary_labels_transform_control: pd.Series, 
                                    data_type_mrd: pd.Series,
                                    data_type_transform_control: pd.Series) -> None:
        '''
            generates list of cluster infos, containing how many test events, 
            control/transform events there are in one cluster and the cluster center
            data= embedding of mrd/control+transform events - used to get the total number of events 
            in one cluster and cluster median for plots
            binary_labels = blast (=1 ) or nonblast (=0)
            data_type = whether it is test/control or transform data CONTROL = 1, TRANSFORM = 2, TEST = 3
            mrd and transfrom-control data is passed seperately to ensure correctness of label order, since 
            hdbscan is fittet first on mrd events and the rest is transfromed at a later stage. -> order is different
            thanin original dataframe
        '''

        if self._predicted_clusters_mrd is None or self._predicted_clusters_transform_control is None:
            raise Exception("ERROR: Data has not been clustered/transformed yet, predicted_clusters is None.\n",
                            "Call cluster_data and transfrom_data functions first.")

        cluster_labels_combined = pd.concat([self._predicted_clusters_mrd, self._predicted_clusters_transform_control],
                                   axis='index',
                                   ignore_index=True)

        data_combined = pd.concat([data_mrd, data_transform_control],
                                   axis='index',
                                   ignore_index=True)

        binary_labels_combined = pd.concat([binary_labels_mrd, binary_labels_transform_control],
                                            axis='index',
                                            ignore_index=True)

        data_type_combined = pd.concat([data_type_mrd, data_type_transform_control],
                                        axis='index',
                                        ignore_index=True)
        cluster_info_list = []

        unqiue_clusters = pd.unique(cluster_labels_combined)

        for cluster_id in unqiue_clusters:
            cluster_info = ClusterInfo()
            current_cluster_mask = cluster_labels_combined == cluster_id
            cluster_info.generate_info_from_cluster_data(cluster_id, 
                                                         data_combined[current_cluster_mask], 
                                                         binary_labels_combined[current_cluster_mask], 
                                                         data_type_combined[current_cluster_mask])

            cluster_info_list.append(cluster_info)

        self._cluster_info = cluster_info_list

    def save_cluster_info(self, base_path: Path, blast_cluster_ids: List):

        cluster_info_dict = {'cluster_id': [],
                                'n_events': [],
                                'n_control': [],
                                'n_control_transform': [],
                                'n_mrd': [],
                                'n_transform': [],
                                'n_blast': [],
                                'n_nonblast': [],
                                'ratio': []
                            }

        for clu_inf in self.get_cluster_info():
            cluster_info_dict['cluster_id'].append(clu_inf.cluster_label)
            cluster_info_dict['n_events'].append(clu_inf.n_events)
            cluster_info_dict['n_control'].append(clu_inf.n_control)
            cluster_info_dict['n_control_transform'].append(clu_inf.n_control_transform)
            cluster_info_dict['n_mrd'].append(clu_inf.n_mrd)
            cluster_info_dict['n_transform'].append(clu_inf.n_transform)
            cluster_info_dict['n_blast'].append(clu_inf.n_blasts)
            cluster_info_dict['n_nonblast'].append(clu_inf.n_non_blasts)
            cluster_info_dict['ratio'].append(clu_inf.ratio)

        cluster_info_df = pd.DataFrame.from_dict(cluster_info_dict)
        cluster_info_df['blast_cluster'] = 0
        cluster_info_df.loc[cluster_info_df['cluster_id'].isin(blast_cluster_ids), 'blast_cluster'] = 1
        cluster_info_df.sort_values(by=['ratio'], 
                                      ascending=False,
                                      axis='index', 
                                      inplace=True)

        filename = self.name + "_cluster-info.csv"
        filepath = base_path / filename
        cluster_info_df.to_csv(filepath, index=False)
        print(f"INFO: successfully saved csv file with cluster info to {filepath}.")

    def save(self):

        fileName = self.name + "-hdbscan.pkl"
        fileWriter = PickleFileWriter(os.path.join(self.basepath, fileName))
        fileWriter.writePickleToOutputfile(self)


    
    
