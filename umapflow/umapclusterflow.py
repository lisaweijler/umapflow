from umapflow.control_data import DataType
from umapflow.umap import UMAPModel
from umapflow.clustering import HDBSCANModel
from umapflow.utils.multiclasslabelshandler import multi_class_labels_to_binary_labels
from umapflow.parse_config import ConfigParser
from umapflow.visualization import UMAPPlotData, UMAPPlot, HDBSCANPlot, PredictionPlot

from typing import Dict, Tuple, List
from pathlib import Path
import pandas as pd
import numpy as np






class UMAPClusterFlow:
    '''
    Class that fits umap, clusters and predicts and creates plots
    Brings functionality together but doesnt hold data.
    
    '''

    def __init__(self, config: ConfigParser) -> None:
        self.multi_class_gates = config['trainer']['_multi_class_gates']
        self.config = config
        self.panel = config["panel"]
        self.base_path = config.save_dir
        self.base_path_plot = Path(config.save_dir) / 'plot'
        self.base_path_model = Path(config.save_dir) / 'model'
        self.base_path_clusterinfo = Path(config.save_dir) / 'clusterinfo'
        self.base_path_model.mkdir(exist_ok=True)  # create if not there yet (first time called) otherwise ignore and dont overwrite
        self.base_path_plot.mkdir(exist_ok=True)
        self.base_path_clusterinfo.mkdir(exist_ok=True)
 
    def transform_umap(self, umap: UMAPModel, transform_data_labels: pd.DataFrame) -> pd.DataFrame:
        
        try:
            _ = umap.get_transformed_data()
            print("INFO: UMAP has transformed data already stored -> using that..")
        except:
            print("INFO: No transformed data found -> transforming now & saving umap..")
            data_to_transform = transform_data_labels.drop(columns=['data_type', 'labels']).to_numpy()
            umap.projectData(data_to_transform)
            umap.save()

        transform_combined_data_labels = pd.concat([transform_data_labels, umap.control_test_data_labels.drop(columns=['umap_labels'])],
                                                    axis='index',
                                                    ignore_index=True)

        transform_projection_embedding_data_labels = pd.concat([pd.DataFrame(umap.get_transformed_data()), pd.DataFrame(umap.get_embedding())],
                                                    axis='index',
                                                    ignore_index=True)

        # works since embedding of umap should have same order of events as the combined_data_labels
        transform_projection_embedding_data_labels['labels'] = transform_combined_data_labels['labels']
        transform_projection_embedding_data_labels['data_type'] = transform_combined_data_labels['data_type']


        return transform_projection_embedding_data_labels

    def create_umap_plots(self, umap: UMAPModel, data_labels: pd.DataFrame, mrd_name: str):
        # umap is needed for axis scale
        # value passed has datatype columns and labels.. 
        # create plots
        # 5 plots control, mrd, control-mrd, transform-control, transform-control-mrd
        # figure/umapmodel name to be saved under
        sample_name = f"{mrd_name} - {self.panel}"
    
        filepaths = [self.base_path_plot / f"{sample_name} - CONTROL-MRD.png",
                     self.base_path_plot / f"{sample_name} - CONTROL.png",
                     self.base_path_plot / f"{sample_name} - MRD.png",
                     self.base_path_plot / f"{sample_name} - CONTROL-TRANSFORM-MRD.png",
                     self.base_path_plot / f"{sample_name} - CONTROL-TRANSFORM.png"]

        captions = [f"{sample_name} - CONTROL-MRD.png",
                    f"{sample_name} - CONTROL.png",
                    f"{sample_name} - MRD.png",
                    f"{sample_name} - CONTROL-TRANSFORM-MRD.png",
                    f"{sample_name} - CONTROL-TRANSFORM.png"]

        plot_data_masks = [(data_labels['data_type'].isin([DataType.CONTROL.value, DataType.TEST.value])),
                           (data_labels['data_type'].isin([DataType.CONTROL.value])),
                           (data_labels['data_type'].isin([DataType.TEST.value])),
                           (data_labels['data_type'].isin([DataType.CONTROL.value, DataType.TRANSFORM.value, DataType.TEST.value])),
                           (data_labels['data_type'].isin([DataType.CONTROL.value, DataType.TRANSFORM.value]))]
                         
        for i in range(len(filepaths)):
            umap_data = UMAPPlotData(umap=umap,
                                     plot_data=data_labels.drop(columns=['labels', 'data_type'])[plot_data_masks[i]].reset_index(drop=True),
                                     labels=data_labels['labels'][plot_data_masks[i]].reset_index(drop=True))
            umap_data.setDefaultColoring(self.multi_class_gates)
            umap_plot = UMAPPlot(filepaths[i],  # filepath
                                 captions[i],                       # caption
                                 umap_data,
                                 point_types=data_labels['data_type'][plot_data_masks[i]].reset_index(drop=True),
                                 use_new_ax=True,
                                 overwrite=self.config['overwrite']
                                )

            umap_plot.generatePlotFile()

    def fit_umap(self, combined_data_labels: pd.DataFrame, mrd_name: str, umap_args: Dict) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:

        # figure/umapmodel name to be saved under
        sample_name = f"{mrd_name} - {self.panel}"

        # labels for semi-supervised umap
        combined_data_labels['umap_labels'] = combined_data_labels['labels']
        combined_data_labels.loc[combined_data_labels['data_type']==DataType.TEST.value, 'umap_labels'] = -1

         # check if model already exists -> load model and give embedding
        model_pickle = self.base_path_model / (sample_name + '-umap.pkl')
        if model_pickle.exists():
            print("INFO: Umap model already exists, loading..")
            umap = UMAPModel.load(model_pickle)
            
        else:  # create model and fit
            print("INFO: Umap model is in the making, fitting data..")
            umap = UMAPModel(sample_name,
                             self.base_path_model,
                             combined_data_labels, # saved here to be used when loade from pickle
                             n_components=umap_args['n_components'],
                             n_neighbors=umap_args['n_neighbors'],
                             op_mix_ratio=umap_args['opt_mix_ratio'],
                             min_dist=umap_args['min_dist'],
                             metric=umap_args['metric'],
                             semi_supervised=umap_args['semi_supervised'])
            # do the umap magic
            events = combined_data_labels.drop(columns=['data_type', 'labels', 'umap_labels']).to_numpy()
            target = combined_data_labels['umap_labels'].to_numpy()
            umap.fitUmap(events, target=target)
            print("INFO: saving umap to pickle file..")
            umap.save()


        return umap

    def create_cluster_plot(self, clusterer: HDBSCANModel, data_labels, mrd_name: str):
        '''
            create 2 plots:
                - one of the clusters of just mrd data, that was used for fitting
                - one of all events with cluster info, and mrd-control-transform data
            
        '''
        cluster_labels_mrd = clusterer.get_predicted_clusters_mrd()
        cluster_labels_combined = pd.concat([clusterer.get_predicted_clusters_mrd(), clusterer.get_predicted_clusters_transform_control()],
                                   axis='index',
                                   ignore_index=True)

        # need to split it up and add control-transform on bottom of mrd events to fit cluster labels
        data_mrd = data_labels.drop(columns=['labels', 'data_type'])[data_labels['data_type'].isin([DataType.TEST.value, DataType.CONTROL.value])].reset_index(drop=True)
        data_transform_control = data_labels.drop(columns=['labels', 'data_type'])[data_labels['data_type'].isin([DataType.TRANSFORM.value])].reset_index(drop=True)
        data_combined = pd.concat([data_mrd, data_transform_control],
                                   axis='index',
                                   ignore_index=True)

        sample_name = f"{mrd_name} - {self.panel}"
    
        filepaths = [self.base_path_plot / f"{sample_name} - HDBSCANclusters-MRD-Control.png",
                     self.base_path_plot / f"{sample_name} - HDBSCANclusters.png"]

        captions = [f"{sample_name} - HDBSCANclusters-MRD-Control",
                    f"{sample_name} - HDBSCANclusters"]

        plot_data = [data_mrd,
                     data_combined]

        clusterinfos = [[],                             # dont use cluster info for mrd cluster plot
                        clusterer.get_cluster_info()]   # normal clusterinfos for all events

        clusterlabels = [cluster_labels_mrd,
                         cluster_labels_combined]


        for i in range(len(filepaths)):
            
            clustererplot = HDBSCANPlot(filepaths[i],  
                                        captions[i],                       
                                        clusterlabels[i],              
                                        plot_data[i],                              # data to plot pd.DAtaframe
                                        clusterer.min_cluster_size, 
                                        clusterer.min_samples,
                                        cluster_info=clusterinfos[i],           # List[ClusterInfo]    
                                        use_new_ax=True,
                                        overwrite=self.config['overwrite'])
            clustererplot.generatePlotFile()

    def cluster(self, data_labels, clusterer_args, mrd_name: str):
        '''
        For now this is hdbscan clustering.
        checks if model already exists, if not, create model and cluster
        data_labels: pd.dataframe projection of mrd, transform and control data with labels and data_type columns
        clustere_args: hdbscan_args of config
        mrd_name: name of sample being processed
        '''

        # figure/umapmodel name to be saved under
        sample_name = f"{mrd_name} - {self.panel}"
        # check if clusterer already exists -> load clusterer and give clustering
        clusterer_pickle = self.base_path_model / (sample_name + '-hdbscan.pkl')
        if clusterer_pickle.exists():
            print("INFO: Clusterer model already exists, loading..")
            clusterer = HDBSCANModel.load(clusterer_pickle)
            
        else:  # create model and fit
            print("INFO: Clusterer model is in the making, clustering data..")
            min_samples = None
            transform_strength = None
            if "min_samples" in clusterer_args:
                min_samples = clusterer_args["min_samples"]
            if "transform_strength" in clusterer_args:
                transform_strength = clusterer_args["transform_strength"]
            clusterer = HDBSCANModel(sample_name,
                             self.base_path_model,
                             min_cluster_size=clusterer_args['min_cluster_size'],
                             min_samples=min_samples,
                             transform_strength=transform_strength,
                             data_type=self.panel,
                             overwrite=self.config['overwrite'])

            # do the cluster magic
            # cluster only on mrd sample and then transform others into it
            embedded_events = data_labels.drop(columns=['data_type', 'labels'])
            embedded_events_mrd_control = embedded_events[data_labels['data_type'].isin([DataType.TEST.value, DataType.CONTROL.value])].reset_index(drop=True)
            embedded_events_transform = embedded_events[data_labels['data_type']==DataType.TRANSFORM.value].reset_index(drop=True)

            clusterer.cluster_data(embedded_events_mrd_control)
            clusterer.transform_data(embedded_events_transform)
            print("INFO: generating cluster info..")

            # generate binary labels
            binary_labels = multi_class_labels_to_binary_labels(data_labels['labels'], self.multi_class_gates)
            binary_labels_mrd_control = binary_labels[data_labels['data_type'].isin([DataType.TEST.value,DataType.CONTROL.value])].reset_index(drop=True)
            binary_labels_transform = binary_labels[data_labels['data_type']==DataType.TRANSFORM.value].reset_index(drop=True)

            # data types pd.Series
            data_type_mrd_control = data_labels.loc[data_labels['data_type'].isin([DataType.TEST.value,DataType.CONTROL.value]), 'data_type'].reset_index(drop=True)
            data_type_transform = data_labels.loc[data_labels['data_type']==DataType.TRANSFORM.value, 'data_type'].reset_index(drop=True)

            clusterer.generate_cluster_info(embedded_events_mrd_control,
                                            embedded_events_transform,
                                            binary_labels_mrd_control,
                                            binary_labels_transform,
                                            data_type_mrd_control,
                                            data_type_transform) 
            print("INFO: saving clusterer model to pickle file..")
            clusterer.save()


        return clusterer


    def get_blast_cluster_ids(self, clusterer: HDBSCANModel, ratio_threshold=0.95) -> List[int]:

       
        cluster_info_list = clusterer.get_cluster_info()
        blast_cluster_ids = []
        # perhaps remove -1 noise? -yes
        for cluster_info in cluster_info_list:
            ratio = cluster_info.ratio
            cluster_id = cluster_info.cluster_label
            
            if ratio >= ratio_threshold and cluster_id != -1: 
                blast_cluster_ids.append(cluster_id)

        # write clusterinfo and cluster prediction to csv
        clusterer.save_cluster_info(self.base_path_clusterinfo, blast_cluster_ids)
        return blast_cluster_ids


    def create_target_output(self, cluster_labels_mrd_control: pd.Series, blast_cluster_ids: List,  data_labels: pd.DataFrame):

        
        test_control_data_mask = data_labels['data_type'] != DataType.TRANSFORM.value
        data_mrd_control = data_labels[test_control_data_mask].reset_index(drop=True)
        test_data_mask = data_mrd_control['data_type'] == DataType.TEST.value

        test_cluster_labels = cluster_labels_mrd_control[test_data_mask]
        multi_class_target = data_mrd_control['labels'][test_data_mask]

        binary_target = multi_class_labels_to_binary_labels(multi_class_target, self.multi_class_gates).to_numpy()
        binary_output = np.zeros(binary_target.shape)
        binary_output[test_cluster_labels.isin(blast_cluster_ids)] = 1

        return binary_target, binary_output

    def create_prediction_plot(self, pred_binary_labels: np.ndarray, data_labels: pd.DataFrame, mrd_name: str, f1_score: float, threshold=None):
        sample_name = f"{mrd_name} - {self.panel}"
        if threshold is not None:
            sample_name = sample_name + "-thr"+ str(threshold)
        filepath = self.base_path_plot / f"{sample_name} - Prediction.png"
        caption = f"{sample_name} - fscore: {f1_score}"
        test_data_labels = data_labels[data_labels['data_type']==DataType.TEST.value]
        test_data = test_data_labels.drop(columns=['data_type','labels'])
        plot = PredictionPlot(filepath,
                               caption,
                               pred_binary_labels,
                               test_data,           # data to plot pd.DAtaframe
                               overwrite=self.config['overwrite'])

        plot.generatePlotFile()

    def create_csv(self, clusterer: HDBSCANModel, pred_binary_labels: np.ndarray, data_labels: pd.DataFrame, mrd_name: str, threshold=None):
        # carefule here mrd is actually mrd+control and trsnform_control only transform, didnt change naming.. need to do that to avoid confusion
        cluster_labels_combined = pd.concat([clusterer.get_predicted_clusters_mrd(), clusterer.get_predicted_clusters_transform_control()],
                                   axis='index',
                                   ignore_index=True)

        # need to split it up and add control-transform on bottom of mrd events to fit cluster labels
        data_mrd = data_labels[data_labels['data_type'].isin([DataType.TEST.value, DataType.CONTROL.value])].reset_index(drop=True)
        data_transform_control = data_labels[data_labels['data_type'].isin([DataType.TRANSFORM.value])].reset_index(drop=True)
        data_combined = pd.concat([data_mrd, data_transform_control],
                                   axis='index',
                                   ignore_index=True)
        data_combined['cluster_labels'] = cluster_labels_combined
        data_combined['prediction'] = -1
        data_combined['prediction'][data_combined['data_type']==DataType.TEST.value] = pred_binary_labels

        sample_name = f"{mrd_name} - {self.panel}"
    
        filepath = self.base_path / f"{sample_name} - data.csv"
        if threshold is not None:
            filepath = self.base_path / f"{sample_name}_thr{threshold} - data.csv"
        data_combined.to_csv(filepath, index=False)





   

    
