from typing import Tuple


def getOuterAxislimit(limits1 : Tuple[int,int], limits2 : Tuple[int,int]):
    return (min(limits1[0], limits2[0]), max(limits1[1], limits2[1]))
