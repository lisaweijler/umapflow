import pathlib
from umapflow.data_loader.transformation_config import TransformationConfig
from typing import List

import numpy as np
from umapflow.data_loader.flowsample import FlowSample


class PreloadedFlowSample(FlowSample):

    def __init__(self, file_path: pathlib.Path) -> None:
        super(PreloadedFlowSample, self).__init__(file_path)

    def load_raw_data(self, markers : List[str], multi_class_gates: List[str], tc: TransformationConfig) -> dict:
        '''
        Loads preloaded tensor. data is matrix and labels is vector. the labels vector can have more than nonblast, blast labels, 
        like for the umap approach (monocytes, granulocytes, etc..). in the info.txt it should be specified, what markers were used and the legend
        for the labels. 
        '''

        data = np.load(str(self.file_path) + '.npy')
        labels = np.load(str(self.file_path) + '_y.npy')

        return {'data': data, 'labels': labels}
