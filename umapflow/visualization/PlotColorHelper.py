import pandas as pd
from typing import List


class ColorManager:

    @staticmethod
    def create_label_string(labels: pd.Series, label_names: List[str]) -> List[str]:
        '''
        Converts label vector [0, 2,2,0,0,1,1,3,...,4,4,2,1,0] to [monocytes, others, granulocytes, blastother,...]
        '''
        
        label_pos_dict = {}
        for i, label_name in enumerate(label_names):
            label_pos_dict[i] = label_name

        result = [label_pos_dict[label] for index, label in labels.items()]

        return result

    @staticmethod
    def get_colors_of_default_labels(label_names: List[str]) -> List[str]:
        color_dict = {
            "promy": "blue",
            "proery": "red",
            "monocytes": "green",
            "granulocytes": "brown",
            "blast34": "black",
            "blastother": "purple",
            "cd34normal": "gold",
            "bermude": "lightcyan",
            "cd34total": "wheat",
            "other": "silver",
            "Other": "silver"
        }

        additional_colors = ["pink", "cyan", "orange"]

        result: List[str] = []

        for label_name in label_names:
            if label_name in color_dict.keys():
                result.append(color_dict[label_name])
            elif len(additional_colors) > 0:
                result.append(additional_colors[-1])
                additional_colors.pop()
            else:
                raise ValueError("no more colors available")

        return result
