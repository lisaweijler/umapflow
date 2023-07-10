from umapflow.utils.picklefilehandlers import PickleFileLoader, PickleFileWriter
from umapflow.utils.executiontimer import executiontimer

import numpy as np
import pandas as pd
import os
from pathlib import Path
from enum import Enum


class DataType(Enum):
    CONTROL = 1
    TRANSFORM = 2
    TEST = 3




class ControlData(object):


    @staticmethod
    def load(filePath: Path):
        if filePath.is_file():  # file --> normal umap
            print("INFO: loading ControlData instance from file: {}".format(filePath))
            fileLoader = PickleFileLoader(filePath)
            return fileLoader.loadPickleFromInputfile()
        else:
            raise FileNotFoundError(f"No control data class instance found at: {filePath}")

    def __init__(self, basepath: Path, config: dict, data_loader):
        super().__init__()
        self.config = config # trainer config
        self.multi_class_gates = self.config['_multi_class_gates']

        self.basepath = basepath
        
        self.control_multiplier = self.config['control_multiplier']
        self.transform_multiplier = self.config['transform_multiplier']
        self.include_adenominator = self.config['include_adenominator']
        self.data_loader = data_loader
        
        self.multi_class_labels_dict = self.create_multi_class_labels_dict()# dict 'name': label, e.g. 'monocytes': 2
        self.data_dict_list = [] # list with data dicts of all train samples including transform, control data


    def create_multi_class_labels_dict(self) -> dict:
        '''
        create dict to map int labels and labels/gate names str
        { 'adenominator': 0, 'monocyotes': ..}
        '''
        label_dict = {}
        for i, label_name in enumerate(self.multi_class_gates):
            label_dict[label_name] = i

        return label_dict
    
    @executiontimer
    def load_control_samples_data(self) -> None:
        '''
        create data matrix of mixed stratified control samples pd.Dataframe
        and list of origninal dicts with sampled transforms and control events np.ndarrays
        could do target as 0 and 1 nonblast and blast in data loader
        {'data': data, 'target': labels, 'labels': labels, 'name': str(file)}

        '''

        if len(self.data_dict_list) != 0:
            print(f'INFO: Data dict list not empty - found {len(self.data_dict_list)} samples -> SKIP loading.')
        
        for sample in self.data_loader:
            # # in dict always numpy ndarrays 
            self.data_dict_list.append(sample) 


    def filter_bermude_cd34total(self, test_data_labels: pd.DataFrame, test_name: str, bermude_path: str, cd34total_path: str):
        # get marker for predicted bermude/cd34total data, can be common or laip/cfu specific, 
        # can be less than for test sample that's why we need it. read form .txt file that is saved in the corresponding path together with the data
        # marker should be the same for bermude and cd34total prediction, but could also be different -> load and check for both seperately
        
        with open(Path(bermude_path) / Path('marker.txt'), 'r') as marker_file:
            infos = marker_file.readlines()
            infos = [x.strip() for x in infos]
            bermude_markers = infos[0].split(',')

        with open(Path(cd34total_path) / Path('marker.txt'), 'r') as marker_file:
            infos = marker_file.readlines()
            infos = [x.strip() for x in infos]
            cd34total_markers = infos[0].split(',')

        # load from specified directory
        bermude_filepath = Path(bermude_path) / test_name
        cd34total_filepath = Path(cd34total_path) / test_name
        bermude_data_labels = pd.DataFrame(np.load(str(bermude_filepath) + '.npy'), columns=bermude_markers)
        bermude_labels = np.load(str(bermude_filepath) + '_y.npy')
        cd34total_data_labels = pd.DataFrame(np.load(str(cd34total_filepath) + '.npy'), columns=cd34total_markers)
        cd34total_labels = np.load(str(cd34total_filepath) + '_y.npy')

        bermude_data_labels['bermude_labels'] = bermude_labels
        cd34total_data_labels['cd34total_labels'] = cd34total_labels

        # check if same numer of events in original and predicted df
        assert len(bermude_data_labels.index) == len(test_data_labels.index) and\
             len(cd34total_data_labels.index) == len(test_data_labels.index)

        
        # merge data to make sure we have right sorting of labels
        bermude_merge_markers = list(set(bermude_markers).intersection(set(test_data_labels.columns)))
        cd34total_merge_markers = list(set(cd34total_markers).intersection(set(test_data_labels.columns)))
        # how='inner' is default -> but should have the same events anyway.. thus same amoutn of rows afterwards
        # test_data_labels_bermude = pd.merge(test_data_labels, bermude_data_labels, on=bermude_merge_markers) 
        # merge is to slow O(n^2)-> join on index faster since pandas uses hashtables for indices
        test_data_labels_bermude = test_data_labels.set_index(bermude_merge_markers).join(bermude_data_labels.set_index(bermude_merge_markers), how='left')
        test_data_labels_bermude.reset_index(inplace=True)
        if len(test_data_labels_bermude.index) != len(test_data_labels.index) or test_data_labels_bermude.isnull().values.any():
            raise ValueError("ERROR: Something went wrong while merging bermude labels to data frame.. please debug me!")

        # test_data_labels_bermude_cd34totalpd.merge(test_data_labels_bermude, cd34total_data_labels, on=cd34total_merge_markers)
        test_data_labels_bermude_cd34total = test_data_labels_bermude.set_index(cd34total_merge_markers).join(cd34total_data_labels.set_index(cd34total_merge_markers), how='left')
        test_data_labels_bermude_cd34total.reset_index(inplace=True)
        if len(test_data_labels_bermude_cd34total.index) != len(test_data_labels.index) or test_data_labels_bermude_cd34total.isnull().values.any():
            raise ValueError("ERROR: Something went wrong while merging cd34total labels to data frame.. please debug me!")

        filtering_mask = (test_data_labels_bermude_cd34total['bermude_labels'] == 1) |\
             (test_data_labels_bermude_cd34total['cd34total_labels'] == 1)
        filtered_test_data_labels = test_data_labels[filtering_mask]
        removed_test_data_labels = test_data_labels[~filtering_mask]

        return filtered_test_data_labels.reset_index(drop=True), removed_test_data_labels.reset_index(drop=True)

    def mix_control_test_events(self, test_data: np.ndarray, test_labels: np.ndarray, 
                                test_name: str, marker: list, 
                                bermude_path: str=None, cd34total_path: str=None) -> pd.DataFrame:
        '''
        This function mixes test sample with the control events, this should then be used to fit UMAP with
        Calculate how many events in total
        Calculate how many events per train sample
        Calculate what percentage to take (equal for each gate specified) to get the number of events for this sample calculated in previous step
        Sample per gate that amount and part it into the right parts for control and transform events -> this makes sure we do not have an overlap of control and transform events
        Append to overal list
        concatenate list and mix with test sample, return control-test event mix and transform mix for this specific test sample
        '''
        test_data_labels = pd.DataFrame(test_data, columns=marker)
        test_data_labels['labels'] = test_labels
        test_data_labels['data_type'] = DataType.TEST.value

        # remove 'adenominator' cells from test-data (for now that is everything not under bermude or cd34total - see config)
        if 'adenominator' not in self.multi_class_gates:
            raise ValueError('Adenominator not in multi_class_gates -> sampling weight for adenominator most likely not specified!')

        removed_test_data_labels = None
        if not self.include_adenominator:
            # remove with GT gates:
            # test_data_labels = test_data_labels[test_data_labels['labels'] != self.multi_class_labels_dict['adenominator']]
            # remove with predicted gates:
            test_data_labels, removed_test_data_labels = self.filter_bermude_cd34total(test_data_labels, test_name, bermude_path, cd34total_path)

        n_events_test = len(test_data_labels.index) # number or events in test sample
        n_events_total = n_events_test * (self.control_multiplier + self.transform_multiplier)
        n_samples_train = len(self.data_dict_list)
        n_events_per_spl = n_events_total / n_samples_train
        control_part = self.control_multiplier / (self.control_multiplier + self.transform_multiplier)
        
        data_labels_control_list = []
        data_labels_transform_list = []
        
        for data_dict in self.data_dict_list:
            spl_wghts = [1.0] * len(self.multi_class_gates)



            data, _ = \
                    self._stratify_multiclassdata(data_dict, sampling_weights=spl_wghts) # get all events of this sample for specified gates
            n_events_multi_classes = data.shape[0]
            weight = min(1.0, n_events_per_spl/n_events_multi_classes)

            for idx, gate in enumerate(self.multi_class_gates):
                weight_gate = weight


                sampling_weights = [0.0] * len(self.multi_class_gates)
                sampling_weights[idx] = weight_gate
                # np.ndarrays are data_gate and labels_gate
                data_gate, labels_gate = \
                    self._stratify_multiclassdata(data_dict, sampling_weights=sampling_weights) # should already be shuffled

                # divide into control and transform -> this makes sure we dont have events double
                n_events_gate_strat = data_gate.shape[0]
                max_idx_control = int(np.floor(n_events_gate_strat * control_part))

                data_labels_gate = pd.DataFrame(data_gate, columns=marker)
                data_labels_gate['labels'] = labels_gate
                data_labels_gate_control = data_labels_gate.iloc[:max_idx_control,:]
                data_labels_gate_transform = data_labels_gate.iloc[max_idx_control:, :]

                # append to general list
                data_labels_control_list.append(data_labels_gate_control.reset_index(drop=True))
                data_labels_transform_list.append(data_labels_gate_transform.reset_index(drop=True))

        control_data_labels = pd.concat(data_labels_control_list, axis='index', ignore_index=True)
        transform_data_labels = pd.concat(data_labels_transform_list, axis='index', ignore_index=True)
        control_data_labels['data_type'] = DataType.CONTROL.value
        transform_data_labels['data_type'] = DataType.TRANSFORM.value
        
        

        combined_data_labels = pd.concat([control_data_labels, test_data_labels], 
                                          axis='index', ignore_index=True)

        # shuffle
        combined_data_labels = combined_data_labels.sample(frac=1, axis='index'
                                                            ).reset_index(drop=True)  # drop=True to not keep old index column

        transform_data_labels = transform_data_labels.sample(frac=1, axis='index'
                                                            ).reset_index(drop=True)  # drop=True to not keep old index column
       
        return combined_data_labels, transform_data_labels, removed_test_data_labels

    def _stratify_multiclassdata(self, data_dict, sampling_weights=None):

        if self.multi_class_gates is None:
            raise ValueError("multi_class_gates can't be None")

        if sampling_weights is None:
            sampling_weights = [1.0] * len(self.multi_class_gates)

        if len(sampling_weights) != len(self.multi_class_gates):
            raise ValueError("there must be  len(multi_class_gates) sampling_weights")

        # convert to datafram as it get loaded as numpy ndarray
        events = pd.DataFrame(data_dict["data"])  
        events['labels'] = data_dict["labels"]

        # class_labels are greedy
        class_labels = list(self.multi_class_labels_dict.values())

        sampled_idx = pd.Index([], dtype='int64')
        # this is still exclusive sampling, since labeling is already exclusive!
        for lab, weight in zip(class_labels, sampling_weights):
            eventsInCustomClass = events[events['labels'] == lab]
            n_sample = round(len(eventsInCustomClass) * weight)
            sampledClassIdx = eventsInCustomClass.sample(n=n_sample, axis="index").index
            sampled_idx = sampled_idx.append(sampledClassIdx)

        sampled_events_labels = events.loc[sampled_idx].reset_index(drop=True)  # drop=True to not keep old index column

        sampled_events = sampled_events_labels.drop(columns='labels').to_numpy()
        sampled_labels = sampled_events_labels['labels'].to_numpy()

        # two np.ndarrays
        return sampled_events, sampled_labels


    def save(self):

        fileName = "ControlData.pkl"
        fileWriter = PickleFileWriter(os.path.join(self.basepath, fileName))
        fileWriter.writePickleToOutputfile(self)

