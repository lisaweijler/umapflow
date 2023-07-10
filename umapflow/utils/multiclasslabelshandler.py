from typing import List
import pandas as pd
import numpy as np



def multi_class_labels_to_binary_labels(multi_class_labels: pd.Series, classes_to_select: List[str]) -> pd.Series:
    """
    convert multiclass labels to just nonblast (0) and blast (1) labels
    """
    blast_labels = []
    for i, label_name in enumerate(classes_to_select):
        if 'blast' in label_name:
            blast_labels.append(i)

    if len(blast_labels) == 0:
        raise Exception("Could not find blast label!")

    # convert multiclass labels to just nonblast (0) and blast (1)
    targets = pd.Series(np.zeros(len(multi_class_labels)))

    for blast_label in blast_labels:
        targets[multi_class_labels.to_numpy() == blast_label] = 1

    return targets
