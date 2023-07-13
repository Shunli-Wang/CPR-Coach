import numpy as np

def binary_precision_recall_curve(y_score, y_true):
    """Calculate the binary precision recall curve at step thresholds.

    Args:
        y_score (np.ndarray): Prediction scores for each class. Shape should be (num_classes, ).
        y_true (np.ndarray): Ground truth many-hot vector. Shape should be (num_classes, ).

    Returns:
        precision (np.ndarray): The precision of different thresholds.
        recall (np.ndarray): The recall of different thresholds.
        thresholds (np.ndarray): Different thresholds at which precision and recall are tested.
    """
    assert isinstance(y_score, np.ndarray)
    assert isinstance(y_true, np.ndarray)
    # print(y_score, y_true)
    assert y_score.shape == y_true.shape

    # make y_true a boolean vector
    y_true = (y_true == 1)
    # sort scores and corresponding truth values, all in decreased order
    desc_score_indices = np.argsort(y_score, kind='mergesort')[::-1]  # get the sort indexs
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    # There may be ties in values, therefore find the `distinct_value_inds`
    distinct_value_inds = np.where(np.diff(y_score))[0]
    threshold_inds = np.r_[distinct_value_inds, y_true.size - 1]
    # accumulate the true positives with decreasing threshold
    tps = np.cumsum(y_true)[threshold_inds]
    fps = 1 + threshold_inds - tps
    thresholds = y_score[threshold_inds]

    precision = tps / (tps + fps)
    precision[np.isnan(precision)] = 0
    recall = tps / tps[-1]
    # stop when full recall attained
    # and reverse the outputs so recall is decreasing
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)

    return np.r_[precision[sl], 1], np.r_[recall[sl], 0], thresholds[sl]

def mean_average_precision(scores, labels):
    """Mean average precision for multi-label recognition.

    Args:
        scores (list[np.ndarray]): Prediction scores of different classes for each sample.
        labels (list[np.ndarray]): Ground truth many-hot vector for each sample.

    Returns:
        np.float: The mean average precision.
    """
    results = []
    scores = np.stack(scores).T  # [14, N]
    labels = np.stack(labels).T  # [14, N]

    # print(scores.shape, labels.shape)

    for score, label in zip(scores, labels):
        precision, recall, _ = binary_precision_recall_curve(score, label)
        ap = -np.sum(np.diff(recall) * np.array(precision)[:-1])
        results.append(ap)
    results = [x for x in results if not np.isnan(x)]
    if results == []:
        return np.nan
    return np.mean(results)

def mmit_mean_average_precision(scores, labels):
    """Mean average precision for multi-label recognition. Used for reporting
    MMIT style mAP on Multi-Moments in Times. The difference is that this
    method calculates average-precision for each sample and averages them among
    samples.

    Args:
        scores (list[np.ndarray]): Prediction scores of different classes for each sample.
        labels (list[np.ndarray]): Ground truth many-hot vector for each sample.

    Returns:
        np.float: The MMIT style mean average precision.
    """
    results = []
    # scores = np.stack(scores).T  # [14, N]
    # labels = np.stack(labels).T  # [14, N]
    scores = np.array([i.numpy() for i in scores])
    labels = np.array([i.numpy() for i in labels])

    for score, label in zip(scores, labels):
        precision, recall, _ = binary_precision_recall_curve(score, label)
        ap = -np.sum(np.diff(recall) * np.array(precision)[:-1])
        results.append(ap)
    return np.mean(results)

def eval_mAP_mmitmAP(scoreList, labelList):
    mAP = mean_average_precision(scoreList, labelList)
    mmit_mAP = mmit_mean_average_precision(scoreList, labelList)
    print('mAP: %.4f, mmit_mAP: %.4f' % (mAP, mmit_mAP))
