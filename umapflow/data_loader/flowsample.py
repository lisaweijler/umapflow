from umapflow.data_loader.transformation_config import TransformationConfig

from pathlib import Path
from typing import Dict, List
import numpy as np
import abc




class FlowSample(abc.ABC):


    FILTER_OBVIOUS_NONBLAST = False  

    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path

    @abc.abstractmethod
    def load_raw_data(self, markers: List[str], multi_class_gates: List[str], transformation_config: TransformationConfig) -> Dict:
        pass

    def load_data(self, markers: List[str], multi_class_gates: List[str], transformation_config: TransformationConfig) -> Dict:

        if not isinstance(transformation_config, TransformationConfig):
            raise TypeError("transformation_config must be from type TransformationConfig")

        if transformation_config == None:
            raise ValueError("transformation config must be set")

        # load raw data
        data_dict = self.load_raw_data(markers, multi_class_gates, transformation_config)

        # transform data
        data_dict_transformed = self._transform(data_dict, transformation_config)
        return data_dict_transformed


    def _transform(self, data_dict: Dict[str,np.ndarray], tc: TransformationConfig) -> Dict:
        """
            - cutoff: the data is cut off if sample size exceeds specified sequencelength

        data and labels are np.ndarrays, 
        """
        data = data_dict['data']
        labels = data_dict['labels']



        if tc.cutoff:
            if data.shape[0] > tc.sequence_length:
                data = data[:tc.sequence_length, :]
                labels = labels[:tc.sequence_length]



        return {'data': data, 'labels': labels}
