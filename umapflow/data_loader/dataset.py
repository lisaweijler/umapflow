from umapflow.data_loader.transformation_config import TransformationConfig
from umapflow.data_loader.preloaded_flowsample import PreloadedFlowSample
from umapflow.data_loader.fm_flowsample import FmFlowSample


from umapflow.utils.util import get_project_root
from shutil import rmtree
from pathlib import Path
from typing import Dict
import numpy as np
from tqdm import tqdm


class FlowData():
    """
    Flow cytometry dataset. The experiment data is stored in xml and fcs files. The fcs files contain the cell data while the xml
    files contain informations regarding the specific experiment. if fast_preload is true during initialization of the object the fcs data
    is preloaded and saved as pickled numpy arrays in the folder determined by fast_preload_dir. The folder is deleted when __del__ ist called.
    """

    def __init__(self, **kwargs):

        self._sanity_check(**kwargs)

        ## set class variables
        self.current_pos = 0
        self.data_type = kwargs['data_type']
        self.marker_list = kwargs['markers']
        self.sequence_length = kwargs['sequence_length']
        self.shuffle = kwargs['shuffle']
        self.cutoff = kwargs['cut-off']  # if true cut samples that are bigger than sequence length
        
        # load only bermude gate for example
        if 'eventtype' in kwargs: 
            self.eventtype = kwargs['eventtype']
        else:
            self.eventtype = 'all'
        self.fast_preload = kwargs['fast_preload']
        self.load_raw_data = "load_raw_data" in kwargs and kwargs["load_raw_data"] == True

        # save multi class gate labels (e.g. monocytes, blast34, granulocytes..)
        if 'multi_class_gates' in kwargs:
            self.multi_class_gates = kwargs['multi_class_gates']
            # dictionary is used for info.txt file
            self.label_legend = {'adenominator':0}
            #multiclass gates is greedy!!:)
            for i, g in enumerate(self.multi_class_gates):
                i += 1
                self.label_legend[g] = i
        else:
            # nothing specified - have labels 0 for non blast and 1 for blasts
            self.multi_class_gates = []
            self.label_legend = {'adenominator': 0, 'blast': 1}

        ## get path to file list
        self.data_root = get_project_root() / Path(kwargs['data_dir'])
        self.data_list_path = self.data_root / Path(kwargs['data_type'] + '.txt')


        ## create transform configs
        self.loading_transformation_config = TransformationConfig(self.cutoff,
                                                                  self.shuffle, self.sequence_length, 
                                                                  self.eventtype)
        self.preloading_transformation_config = TransformationConfig(False, 
                                                                     self.shuffle, self.sequence_length,
                                                                     self.eventtype)

        # Preload files or get list of filepaths if already preloaded
        if self.fast_preload:
            pre_load_dir = Path(kwargs['fast_preload_dir'])
            pre_load_dir.mkdir(parents=False, exist_ok=True)
            self.tmp_path = pre_load_dir / Path(self.data_type)
            self.files = self._preload()
        else:
            self.files = self._get_filepath_list()

        print('Created {} dataset with {} files from {}.'.format(
            kwargs['data_type'], len(self.files), self.data_list_path))
        print('')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx) -> Dict:
        file = self.files[idx]

        # if fast_preload is turned on directly load arrays otherwise load from fcs files
        if self.fast_preload:
            flowsample = PreloadedFlowSample(file)
        else:
            flowsample = FmFlowSample(file)

        if self.load_raw_data:
            data_dict = flowsample.load_raw_data(self.marker_list, self.multi_class_gates, self.loading_transformation_config)
        else:
            data_dict = flowsample.load_data(self.marker_list, self.multi_class_gates, self.loading_transformation_config)

        data = data_dict["data"]
        labels = data_dict["labels"]

        
        return {'data': data, 'target': labels, 'labels': labels, 'name': str(file)}

    def __iter__(self):
        self.current_pos = 0
        return self

    def __next__(self) -> Dict:

        if self.current_pos >= len(self):
            raise StopIteration

        data = self[self.current_pos]
        self.current_pos += 1
        return data

    def _sanity_check(self, **kwargs):
        # sanity checks
        assert kwargs['data_type'] in ['train', 'eval', 'test'], \
            'data_type must be either "train", "eval" or "test" but got {}'.format(
                kwargs['data_type'])


    def _get_filepath_list(self):
        """
        Get list of experiment files (*.xml and *.analysis) from list self.data_list_path.
        """
        with open(self.data_list_path, 'r') as file:
            files = file.readlines()
        files = [f.strip() for f in files]
        files = [Path(f) for f in files]# if Path(f).is_file()]

        return files

    def _preload(self):
        """
        Preloads the dataset and stores the numpy tensors directly to save loading time.
        if gates is not none then save in the labels vector not only 0 and 1 for blast non blast but also all other gates
        specified in the gates list and write legend in info.txt. f.e. 1 = monocytes, 2 = granulocytes, 3 = proery, 4 = blasts, 0 = rest. 
        """
        # If already loaded, dont preload again and set load = False
        if self.tmp_path.is_dir():
            # By storing the data as np arrays  we lose the column names and label legend. To make sure that the preloaded files
            # match to the markers and labels we chose for the current experiment we save a text file with the marker list and chosen labels
            # of the preloaded file. In theory we could also just preload the file with all the markers and afterwards
            # select the correct parts of the numpy tensor but just enforcing the same list of markers as is done
            # here is simpler and less error prone.

            try:
                with open(self.tmp_path / Path('info.txt'), 'r') as info_file:
                    infos = info_file.readlines()
                    infos = [x.strip() for x in infos]

                    preloaded_markers = infos[0].split(',')
                    label_legend = dict()
                    for i in range(1,len(infos)):
                        # e.g. key = "monocytes", label = 2
                        key, label = infos[i].split(':')
                        key = key.strip()
                        label = float(label.strip())
                        label_legend[key] = label

            except FileNotFoundError:
                preloaded_markers = []
                label_legend = dict()

            # if markers do not match just preload again ...
            if preloaded_markers != self.marker_list or label_legend != self.label_legend:
                rmtree(self.tmp_path)

                print('Existing tmp folder was created with markers = {} and label_legend = {}. Preload fcs files again in {}.'
                      .format(", ".join(self.marker_list), self.label_legend, self.tmp_path))
                load = True
                self.tmp_path.mkdir(parents=False, exist_ok=True)
            else:
                print('Preloaded fcs files already exist in {}.'.format(self.tmp_path))
                load = False
        else:
            print('Preload fcs files in {}.'.format(self.tmp_path))
            load = True
            self.tmp_path.mkdir(parents=False, exist_ok=True)

        # Load files and save as numpy arrays
        exp_files = self._get_filepath_list()
        file_descriptor_list = []

        for exp in tqdm(exp_files):

            # file_descriptors are here the file names of the 'data' tensor without the '.npa' ending.
            # This way the data tensor is file_descriptor + '.npa' and the label tensor is file_descriptor + '_y.npa
            file_descriptor = self.tmp_path / str(exp.stem)
            file_descriptor_list.append(file_descriptor)

            if load:
                flowsample = FmFlowSample(exp)
                exp_tensor_dict = flowsample.load_data(self.marker_list, self.multi_class_gates, self.preloading_transformation_config)

                # convert pd df to numpy and save as numpy.ndarray
                np.save(str(file_descriptor) + '.npy', exp_tensor_dict['data']) # uses pickle careful with different python envs etc
                np.save(str(file_descriptor) + '_y.npy', exp_tensor_dict['labels']) # uses pickle

        with open(self.tmp_path / Path('info.txt'), 'w') as info_file:
            marker_str = ','.join(self.marker_list) + '\n'
            info_file.write(marker_str)
            for k, v in self.label_legend.items():
                info_file.write(k + ': ' + str(v) + '\n')

        return file_descriptor_list
