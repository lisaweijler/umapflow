import argparse
import pandas as pd
import numpy as np
import random
import time
from pathlib import Path

from umapflow import UMAPClusterFlow
from umapflow.control_data import ControlData
from umapflow.utils import MetricTracker
from umapflow.parse_config import ConfigParser
from umapflow.utils.multiclasslabelshandler import multi_class_labels_to_binary_labels
import umapflow

random.seed(42)



def main(config):


    # setup data_loader instances
    
    test_data_set = config.init_obj(
        'data_loader', umapflow, data_type='test')

    # setup data_loader instances
    control_data_set = config.init_obj(
        'data_loader', umapflow, data_type='train')

    

    control_data_path = Path(config.save_dir) / "ControlData.pkl"
    if control_data_path.is_file():
        control_data = ControlData.load(control_data_path)
        control_data.load_control_samples_data()
    else: 
        control_data = ControlData(config.save_dir, config['trainer'], control_data_set)
        control_data.load_control_samples_data()
        control_data.save()

    umapclusterflow = UMAPClusterFlow(config)

    if "use_umap" in config.keys():
        use_umap = config["use_umap"]
    else:
        use_umap = True

    if "min_mrd" in config.keys():
        min_mrd = config["min_mrd"]
    else:
        min_mrd = 0.0

    if "ratio_threshold" in config.keys():
        ratio_threshold = config["ratio_threshold"]
    else:
        ratio_threshold = [0.95]
    # to create several output files, one for each specified threshold
    metric_fns = []
    metrics = []
    for i in range(len(ratio_threshold)):
        metric_fns.append([getattr(umapflow, met) for met in config['metrics']])
        metrics.append(MetricTracker(*[m.__name__ for m in metric_fns[i]]))

    file_list = []
    n_events_list = []
    mrd_gt_adenominator = []
    for mrd_data in test_data_set:
        mrd_name = mrd_data["name"].split('/')[-1]
        print(mrd_name)
        n_events = mrd_data['data'].shape[0]
        binary_labels = multi_class_labels_to_binary_labels(pd.Series(mrd_data['labels']), 
                                                       config['data_loader']['args']['multi_class_gates'])
        n_blasts = binary_labels.sum()
        mrd_gt = n_blasts/n_events
        if mrd_gt < min_mrd:
            # skip sample
            continue

        mrd_gt_adenominator.append(mrd_gt)
        n_events_list.append(n_events)
        file_list.append(mrd_name)
        bermude_path = config['trainer']['bermude_path'] if 'bermude_path' in config['trainer'] else None
        cd34total_path = config['trainer']['cd34total_path'] if 'cd34total_path' in config['trainer'] else None

        combined_data_labels, transform_data_labels, removed_test_data_labels = control_data.mix_control_test_events(mrd_data["data"], mrd_data["labels"], 
                                                                                            mrd_name, config['data_loader']['args']['markers'],
                                                                                            bermude_path=bermude_path, cd34total_path=cd34total_path)
        if use_umap:
            # fit_umap only uses freshly shuffled combined_data_labels if no umap instance was there to laod
            # otherwise use combined_data_labels instance variable that got saved to ensure that labels and events fit togheter
            umap = umapclusterflow.fit_umap(combined_data_labels, mrd_name, config['arch']['args']['umap_args'])
            transform_projection_embedding_data_labels = umapclusterflow.transform_umap(umap, 
                                                                                        transform_data_labels)
            # creates 5 plots: control, mrd, control-mrd, transform-control, transform-control-mrd
            umapclusterflow.create_umap_plots(umap, transform_projection_embedding_data_labels, mrd_name)
        else:

            transform_projection_embedding_data_labels = pd.concat([transform_data_labels, combined_data_labels],
                                                    axis='index',
                                                    ignore_index=True)

        clusterer = umapclusterflow.cluster(transform_projection_embedding_data_labels,
                                            config['arch']['args']['hdbscan_args'],
                                            mrd_name)
        umapclusterflow.create_cluster_plot(clusterer, transform_projection_embedding_data_labels, mrd_name)
        # create results for all thresholds specified

        for idx,thr in enumerate(ratio_threshold):
            blast_cluster_ids = umapclusterflow.get_blast_cluster_ids(clusterer, ratio_threshold=thr)
            binary_target, binary_output = umapclusterflow.create_target_output(clusterer.get_predicted_clusters_mrd(),
                                                                                blast_cluster_ids, 
                                                                                transform_projection_embedding_data_labels)

            binary_target_complete = binary_target
            binary_output_complete = binary_output
            if removed_test_data_labels is not None:
                # added here bc of plot, only cropped sample is plottet but fscore should be calculated with all data
                # all removed events due to bermude/cd34total prediction are predicted to be non blasts
                removed_binary_target = multi_class_labels_to_binary_labels(removed_test_data_labels['labels'], 
                                                                                            config['data_loader']['args']['multi_class_gates']).to_numpy()
                removed_binary_output = np.zeros(removed_binary_target.shape)
                binary_target_complete = np.append(binary_target_complete, removed_binary_target)
                binary_output_complete = np.append(binary_output_complete, removed_binary_output)


            for met in metric_fns[idx]:
                met_value = met(binary_output_complete, binary_target_complete)
                metrics[idx].update(met.__name__, met_value)
                if met.__name__ == "f1_score":
                    fscore = met_value
                    print(f"fscore: {met_value}")

                    umapclusterflow.create_prediction_plot(binary_output, transform_projection_embedding_data_labels, mrd_name, fscore, threshold=thr)

            umapclusterflow.create_csv(clusterer, binary_output, transform_projection_embedding_data_labels, mrd_name, threshold=thr)

    for idx,thr in enumerate(ratio_threshold):
        results_path = config.save_dir / (time.strftime("%Y-%m-%d %H-%M-%S") + '-' + config['name'] + '_'+ str(thr)+ '_test_results.txt')
        results_to_file(metric_fns[idx], metrics[idx], file_list, mrd_gt_adenominator, n_events_list, config['name'], results_path)


  

 
 


def results_to_file(post_metric_fns, post_metrics, file_list, mrd_gt_adenominator, n_events_list, config_name, results_path):

    # Log results for every file

    metric_names_list = [f'{f.__name__}' for f in post_metric_fns]
    metric_data = post_metrics.data()

    log_list = []
    for n, file in enumerate(file_list):
        metric_result_list = [
            f'{metric_data[m.__name__][n]}' for m in post_metric_fns]
        log_list.append(f'{file}, unknown, ' + str(n_events_list[n]) + ', ' + ', '.join(metric_result_list) + ', ' + str(mrd_gt_adenominator[n]))

    header = f"// config_name: {config_name}\n"

    cols = "# experiment, label, total, " + ", ".join(metric_names_list) + ", mrd_gt_adenominator\n"

    with open(results_path, 'w') as text_file:
        text_file.write(header)
        text_file.write(cols)
        # write file results
        for l in log_list:
            text_file.write(l+'\n')


if __name__ == '__main__':
    config_debug = 'config/example_experiment.json'
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=config_debug, type=str,
                      help='config file path (default: None)')

    config = ConfigParser.from_args(args)
    main(config)
