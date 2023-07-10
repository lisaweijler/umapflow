from enum import unique
import enum
import pathlib
from typing import List, Tuple, Dict
#import flowme
import numpy as np
import pandas as pd

from umapflow.data_loader.flowsample_labelnames import GateCollection
from umapflow.data_loader.transformation_config import TransformationConfig
from umapflow.utils.util import suppress_stdout_stderr
from umapflow.data_loader.flowsample import FlowSample





@unique
class EventType(enum.Enum):
    '''
    Event subsets.
    '''
    BLAST = 1
    NONBLAST = 0
    ALL = -1
    CD34TOTAL = 2
    BERMUDE = 3
    BERMUDE_CD34TOTAL = 5


class FmFlowSample(FlowSample):

    FILTER_OBVIOUS_NONBLAST = False  # nees to be further discussed, based on threshhold get rid of events

    def __init__(self, file_path: pathlib.Path) -> None:
        super(FmFlowSample, self).__init__(file_path)

    def _add_intactfilter_to_mask(self, gates, mask):
        if GateCollection.GATE_ADENOMINATOR in gates.columns:
            mask = (mask) & (gates[GateCollection.GATE_ADENOMINATOR] == 1)

        if GateCollection.GATE_INTACT in gates.columns:
            mask = (mask) & (gates[GateCollection.GATE_INTACT] == 1)

            return mask

    def _get_labels(self, events, gates, multi_class_gates: List[str]) -> pd.DataFrame:
        '''
        Creates new column GT (groundtruth) in gates df, with the gt
        '''

        if len(multi_class_gates) == 0:
            # no specific gates specified -> 0 for nonblasts and 1 for blasts
            blast_mask = self._get_blast_mask(events, gates)

            gates['GT'] = blast_mask
            gates['GT'] = gates['GT'].astype(int)  # convert True, False to 1, 0
            #gates.loc[blast_mask, 'GT'] = 1
        else:

            gates["GT"] = 0

            for i, class_gate in enumerate(multi_class_gates):
                if not class_gate in gates.columns:
                    raise ValueError(f"class_gate '{class_gate}' is not present in the data")

                # multi_class_label is greedy!
                mask_events_in_gate = gates[class_gate] > 0.5  # instead of ==1 to avoid bugs of types etc
                gates.loc[mask_events_in_gate, "GT"] = i

        return gates

    def _get_blast_mask(self, events, gates):

        blastLabel = EventType.BLAST.value
        defaultSelection = False  # defaultSelection if no blast Gates are presented
        mask = events.shape[0] * [defaultSelection]

        if GateCollection.GATE_BLAST in gates.columns:
            mask = gates[GateCollection.GATE_BLAST] == blastLabel
        elif GateCollection.GATE_BLASTEN in gates.columns:
            mask = gates[GateCollection.GATE_BLASTEN] == blastLabel

        elif GateCollection.GATE_BLASTOTHER in gates.columns and GateCollection.GATE_BLAST34 in gates.columns:
            mask = (gates[GateCollection.GATE_BLASTOTHER] == blastLabel) | (gates[GateCollection.GATE_BLAST34] == blastLabel)

        elif GateCollection.GATE_BLASTOTHER in gates.columns:
            mask = gates[GateCollection.GATE_BLASTOTHER] == blastLabel
        elif GateCollection.GATE_BLAST34 in gates.columns:
            mask = gates[GateCollection.GATE_BLAST34] == blastLabel

        return mask

    def _get_cd34total_mask(self, events, gates):

        mask = events.shape[0]*[False]  # select all

        if GateCollection.GATE_CD34TOTAL in gates.columns:
            mask = gates[GateCollection.GATE_CD34TOTAL] > 0.5  # hard coded for now.. same as blastlable
        else:
            print(f"WARNING: {GateCollection.GATE_CD34TOTAL} not in sample gates {gates.columns}. No filtering took place.")
        return mask

    def _get_bermude_mask(self, events, gates):

        mask = events.shape[0]*[False]  # select all

        if GateCollection.GATE_BERMUDE in gates.columns:
            mask = gates[GateCollection.GATE_BERMUDE] > 0.5  # hard coded for now.. same as blastlable
        else:
            print(f"WARNING: {GateCollection.GATE_BERMUDE} not in sample gates {gates.columns}. No filtering took place.")
        return mask

    def _get_filtered_data(self, events, gates, data_type: EventType):
        '''
        Filter data based on datatype,
        ALL : get all events from intact/adenominator gate on
        NONBLAST: get all nonblasts from intact/adenominator gate on
        BLAST: get all blasts from intact/adenominator gate on
        CD34TOTAL: get all CD34total from intact/adenominator gate on
        BERMUDE: get all bermude from intact/adenominator gate on
        '''
        mask = events.shape[0] * [True]  # select all

        if data_type != EventType.ALL:
            if data_type == EventType.CD34TOTAL:
                mask = self._get_cd34total_mask(events, gates)
            elif data_type == EventType.BERMUDE:
                mask = self._get_bermude_mask(events, gates)
            elif data_type == EventType.BERMUDE_CD34TOTAL:
                cd34_total_mask = self._get_cd34total_mask(events, gates)
                bermude_mask = self._get_bermude_mask(events, gates)
                mask = (cd34_total_mask) | (bermude_mask)
            else:
                mask = self._get_blast_mask(events, gates)  # this is weird. talk to Florian
                blastLabel = EventType.BLAST.value
                if blastLabel == EventType.NONBLAST:
                    mask = np.invert(mask)

        mask = self._add_intactfilter_to_mask(gates, mask)
        mask = self._add_obviousNonBlastFilter_to_mask(events, mask)

        return events[mask], gates[mask]

    def _add_obviousNonBlastFilter_to_mask(self, events, mask):

        if FlowSample.FILTER_OBVIOUS_NONBLAST:
            mask = (mask) & (events["SSC-A"] < 2.5)

        return mask

    def _rename_markers(self, events: pd.DataFrame):
        renameDict = {
            "CD3BV605": "CD3",
            "CD4AX700": "CD4",
            "CD5PE": "CD5",
            "CD5PC7": "CD5",
            "CD7ECD": "CD7",
            "CD8AX750": "CD8",
            "CD48BL": "CD48",
            "CD56CD16": "CD56CD16",
            "CD56CD16KO": "CD56CD16",
            "CD56+16": "CD56CD16",
            "CD56+CD16": "CD56CD16",
            "CD99APC": "CD99",
            "CD99PE": "CD99",
            "CD312APC": "CD312",
            # until here T-ALL
            "CD13_19": "CD13CD19",
            "CD15-FITC": "CD15",
            "CD117-PC5.5": "CD117",
            "CD33-PC7": "CD33",
            "CD13-APC": "CD13",
            "13": "CD13",
            "CD11B-APC-A750": "CD11B",
            "CD11B-A750": "CD11B",
            "HLA-DR-PB": "HLA-DR",
            "HLADR-PB": "HLA-DR",
            "CD45-KRO": "CD45",  # kein CD35KR? - sollte so passen
            "CD45RA-A750": "CD45RA",
            "CD38-FITC": "CD38",
            "CD99-PE": "CD99",
            "CD371-APC": "CD371",
            "CD123-APC-A700": "CD123",
            "CD45RA-APC-A750": "CD45RA",
            "CD7_19": "CD7CD19",
            "CD7_CD19": "CD7CD19",
            "CD34-ECD": "CD34",
            "CD14-APC-A700": "CD14",
            "CD14-A700": "CD14",
            "CD19+7-PE": "CD19CD7",
            "CD19+CD7-PE": "CD19CD7",
            "CD7_56": "CD7CD56",
            "CD7/56": "CD7CD56",
            "CD19/56": "CD19CD56",
            "CD7-CD13": "CD7CD13",
            "CD7VCD56": "CD7CD56",
            "CD56CD7": "CD7CD56",
            "CD123-A700": "CD123",
            "CLL1-APC": "CD371",
            "CLL1": "CD371",
            "HLADR": "HLA-DR"
        }
        renameDict = {k: v for k, v in renameDict.items() if k in events.columns and not v in events.columns}
        if len(renameDict) > 0:
            events.rename(columns=renameDict, inplace=True)

        return events

    def _filter_for_selected_marker(self, file: str, events: pd.DataFrame, markers: List[str]) -> pd.DataFrame:
        columnsToSelect = markers
        # check if each column exists in the data
        eventCols = events.columns
        for colName in columnsToSelect:
            if not colName in eventCols:
                raise ValueError(f"requested marker '{colName}' is not presented in the data of sample {file}. data columns: {eventCols}")

        return events.loc[:, columnsToSelect]

    def _get_eventtype(self, eventtype: str) -> EventType:

        if eventtype == 'all':
            return EventType.ALL
        if eventtype == 'blast':
            return EventType.BLAST
        if eventtype == 'nonblast':
            return EventType.NONBLAST
        if eventtype == 'cd34total':
            return EventType.CD34TOTAL
        if eventtype == 'bermude':
            return EventType.BERMUDE
        if eventtype == "bermude_cd34total":
            return EventType.BERMUDE_CD34TOTAL

    def load_raw_data(self, markers: List[str], multi_class_gates: List[str], tc: TransformationConfig) -> Dict[str,np.ndarray]:
        """
        Load fcs data from an xml file accompanied with a fcs file. the 'file' input parameter
        is the name of the directory that contains the xml file.
        """
        # Load fcs data

        # NOTE: uncomment the following if you want to use flowme for loading raw fcs files

        # with suppress_stdout_stderr():  # suppress qinfo output
        #     sample = flowme.fcs(str(self.file_path))

        #     # Remove dublicate columns if available
        #     events = sample.events()
        #     if len(set(events.columns)) != len(events.columns):
        #         events = events.loc[:, ~events.columns.duplicated()]

        # # rename marker labels to standard labels
        # # select columns based on specified markers
        # events = self._rename_markers(events)
        # events = self._filter_for_selected_marker(str(self.file_path), events, markers)

        # gates = sample.gate_labels()  # get gating (GT) information

        # # convert eventtype string to EventType
        # eventtype = self._get_eventtype(tc.eventtype)
        # # returns events, gates from intact/adenominator gate on
        # data, gates = self._get_filtered_data(events, gates, eventtype)
        # # _get_labels() returns gates matrix pd.dataframe with an additional column "GT", that defines
        # # which ones are blasts (=1), non blast (=0), or monocytes, granulocytes etc. greedy algorithm
        # labels = self._get_labels(data, gates, multi_class_gates)

        # data, labels = self._rearrange_data(data, labels, tc)

        # # Convert to numpy tensors
        # data = data.to_numpy()
        # labels = labels['GT'].to_numpy()

        # data_dict = {'data': data, 'labels': labels}

        # return data_dict
        
        return None

    def _rearrange_data(self, data: pd.DataFrame, labels: pd.DataFrame, tc: TransformationConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:

        if tc.shuffle:
            data_labels = pd.concat([data, labels], axis=1)
            data_labels = data_labels.sample(frac=1)

            data = data_labels.iloc[:, 0:len(data.columns)]
            labels = data_labels.iloc[:, len(data.columns):]

        return data, labels
