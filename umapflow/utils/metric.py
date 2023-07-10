


def f1_score(output, target, eps=0.00001):
    pred = (output > 0.5).astype(int)
    assert pred.shape == target.shape, f'pred shape {pred.shape} != target shape {target.shape}'

    tp = ((pred == 1)*(target == 1)).sum()
    fp = ((pred == 1)*(target == 0)).sum()
    fn = ((pred == 0)*(target == 1)).sum()

    # if there are no blast cells and none are detected f1_score is one
    if tp + fn == 0 and fp == 0:
        return 1.0
    else:
        return 2*tp/(2*tp + fp + fn + eps)

def tp(output, target):

    pred = (output > 0.5).astype(int)
    assert pred.shape == target.shape, f'pred shape {pred.shape} != target shape {target.shape}'
    tp = ((pred == 1)*(target == 1)).sum()
    
    return tp

def tn(output, target):
    pred = (output > 0.5).astype(int)
    assert pred.shape == target.shape, f'pred shape {pred.shape} != target shape {target.shape}'
    tn = ((pred == 0)*(target == 0)).sum()
    
    return tn

def fp(output, target):
    
    pred = (output > 0.5).astype(int)
    assert pred.shape == target.shape, f'pred shape {pred.shape} != target shape {target.shape}'
    fp = ((pred == 1)*(target == 0)).sum()
    
    return fp

def fn(output, target):

    pred = (output > 0.5).astype(int)
    assert pred.shape == target.shape, f'pred shape {pred.shape} != target shape {target.shape}'
    fn = ((pred == 0)*(target == 1)).sum()
    
    return fn

def precision(output, target, eps=0.00001):
    
    pred = (output > 0.5).astype(int)
    assert pred.shape == target.shape, f'pred shape {pred.shape} != target shape {target.shape}'

    tp = ((pred == 1)*(target == 1)).sum()
    fp = ((pred == 1)*(target == 0)).sum()
    fn = ((pred == 0)*(target == 1)).sum()

    return tp/(tp + fp + eps)


def recall(output, target, eps=0.00001):
    pred = (output > 0.5).astype(int)
    assert pred.shape == target.shape, f'pred shape {pred.shape} != target shape {target.shape}'

    tp = ((pred == 1)*(target == 1)).sum()
    fp = ((pred == 1)*(target == 0)).sum()
    fn = ((pred == 0)*(target == 1)).sum()

    return tp/(tp + fn + eps)




def mrd_gt(output, target, eps=0.00001):

    pred = (output > 0.5).astype(int)
    assert pred.shape == target.shape, f'pred shape {pred.shape} != target shape {target.shape}'

    blast = ((target == 1)).sum()
    other = ((target == 0)).sum()

    return blast/(blast + other)


def mrd_pred(output, target, eps=0.00001):

    pred = (output > 0.5).astype(int)
    assert pred.shape == target.shape, f'pred shape {pred.shape} != target shape {target.shape}'

    blast = ((pred == 1)).sum()
    other = ((pred == 0)).sum()

    return blast/(blast + other)
