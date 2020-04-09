import numpy as np
import matplotlib.pyplot as plt
from tools import read_predicted_boxes, read_ground_truth_boxes


def calculate_iou(prediction_box, gt_box):
    """Calculate intersection over union of single predicted and ground truth box.

    Args:
        prediction_box (np.array of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (np.array of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]

        returns:
            float: value of the intersection of union for the two boxes.
    """
    # YOUR CODE HERE

    # Compute intersection

    # Compute intersection
    overlap = 0
    dx = min(prediction_box[2], gt_box[2]) - max(prediction_box[0], gt_box[0])
    dy = min(prediction_box[3], gt_box[3]) - max(prediction_box[1], gt_box[1])
    if (dx >= 0) and (dy >= 0):
        overlap = dx * dy

    # Compute union
    area_prediction = (prediction_box[2] - prediction_box[0]) * (prediction_box[3] - prediction_box[1])
    area_gt = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
    area_union = area_gt + area_prediction - overlap
    iou = overlap / area_union

    assert iou >= 0 and iou <= 1
    return iou


def calculate_precision(num_tp, num_fp, num_fn):
    """ Calculates the precision for the given parameters.
        Returns 1 if num_tp + num_fp = 0

    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of precision
    """
    if num_tp + num_fp == 0:
        return 1
    else:
        return num_tp / (num_tp + num_fp)


def calculate_recall(num_tp, num_fp, num_fn):
    """ Calculates the recall for the given parameters.
        Returns 0 if num_tp + num_fn = 0
    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of recall
    """
    if num_tp + num_fn == 0:
        return 0
    else:
        return num_tp / (num_tp + num_fn)


def get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold):
    """Finds all possible matches for the predicted boxes to the ground truth boxes.
        No bounding box can have more than one match.

        Remember: Matching of bounding boxes should be done with decreasing IoU order!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
    Returns the matched boxes (in corresponding order):
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of box matches, 4].
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of box matches, 4].
            Each row includes [xmin, xmax, ymin, ymax]
    """
    # Find all possible matches with a IoU >= iou threshold

    # Find all possible matches
    iou = np.zeros((len(prediction_boxes) + 1, len(gt_boxes) + 1))
    iou[-1, :] = np.arange(len(gt_boxes) + 1)
    iou[:, -1] = np.arange(len(prediction_boxes) + 1)
    for i in range(len(prediction_boxes)):
        for j in range(len(gt_boxes)):
            iou[i, j] = calculate_iou(prediction_boxes[i], gt_boxes[j])

    # Sort all matches on IoU in descending order
    matched_prediction_boxes = []
    matched_gt_boxes = []

    while iou.shape[0] > 1 and iou.shape[1] > 1 and iou[:-1, :-1].max() >= iou_threshold:
        i, j = np.where(iou[:-1, :-1] == iou[:-1, :-1].max())
        ind1 = int(iou[i[0], -1])
        ind2 = int(iou[-1, j[0]])
        matched_prediction_boxes.append(prediction_boxes[ind1])
        matched_gt_boxes.append(gt_boxes[ind2])

        # these boxes cannot be matched again, so delete this row and collumn
        iou = np.delete(iou, i[0], 0)
        iou = np.delete(iou, j[0], 1)

    return np.array(matched_prediction_boxes), np.array(matched_gt_boxes)

    # Find all matches with the highest IoU threshold


    
    #return np.array([]), np.array([])


def calculate_individual_image_result(prediction_boxes, gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes,
       calculates true positives, false positives and false negatives
       for a single image.
       NB: prediction_boxes and gt_boxes are not matched!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
    Returns:
        dict: containing true positives, false positives, true negatives, false negatives
            {"true_pos": int, "false_pos": int, false_neg": int}
    """

    matched_prediction_boxes, matched_gt_boxes = get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold)
    num_tp = len(matched_prediction_boxes)
    num_fp = len(prediction_boxes) - len(matched_prediction_boxes)
    num_fn = len(gt_boxes) - len(matched_gt_boxes)

    return {"true_pos": num_tp, "false_pos": num_fp, "false_neg": num_fn}


def calculate_precision_recall_all_images(
all_prediction_boxes, all_gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates recall and precision over all images
       for a single image.
       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
    Returns:
        tuple: (precision, recall). Both float.
    """
    num_tp = 0
    num_fp = 0
    num_fn = 0

    for i in range(len(all_prediction_boxes)):
        dict = calculate_individual_image_result(all_prediction_boxes[i], all_gt_boxes[i], iou_threshold)
        num_tp += dict["true_pos"]
        num_fp += dict["false_pos"]
        num_fn += dict["false_neg"]

    precision = calculate_precision(num_tp, num_fp, num_fn)
    recall = calculate_recall(num_tp, num_fp, num_fn)

    return (precision, recall)


def get_precision_recall_curve(
    all_prediction_boxes, all_gt_boxes, confidence_scores, iou_threshold
):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates the recall-precision curve over all images.
       for a single image.

       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        scores: (list of np.array of floats): each element in the list
            is a np.array containting the confidence score for each of the
            predicted bounding box. Shape: [number of predicted boxes]

            E.g: score[0][1] is the confidence score for a predicted bounding box 1 in image 0.
    Returns:
        precisions, recalls: two np.ndarray with same shape.
    """
    # Instead of going over every possible confidence score threshold to compute the PR
    # curve, we will use an approximation
    confidence_thresholds = np.linspace(0, 1, 500)
    precisions = []
    recalls = []

    # only use boxes with confidence > threshold for calculation of precision and recalls
    for confidence_threshold in confidence_thresholds:
        prediction_boxes_used = []
        for i in range(len(all_prediction_boxes)):
            prediction_boxes_used.append([])
            for j in range(len(all_prediction_boxes[i])):
                if confidence_scores[i][j] > confidence_threshold:
                    prediction_boxes_used[i].append(all_prediction_boxes[i][j])
        precision, recall = calculate_precision_recall_all_images(
            prediction_boxes_used, all_gt_boxes, iou_threshold)
        precisions.append(precision)
        recalls.append(recall)

    return np.array(precisions), np.array(recalls)

def plot_precision_recall_curve(precisions, recalls):
    """Plots the precision recall curve.
        Save the figure to precision_recall_curve.png:
        'plt.savefig("precision_recall_curve.png")'

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        None
    """
    plt.figure(figsize=(20, 20))
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.8, 1.0])
    plt.ylim([0.8, 1.0])
    plt.savefig("precision_recall_curve.png")


def calculate_mean_average_precision(precisions, recalls):
    """ Given a precision recall curve, calculates the mean average
        precision.

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        float: mean average precision
    """
    # Calculate the mean average precision given these recall levels.
    recall_levels = np.linspace(0, 1.0, 11)
    # YOUR CODE HERE

    max_precisions = []
    for recall_level in recall_levels:
        ind = np.where(recalls >= recall_level)
        if len(ind[0]) > 0:
            max_precisions.append(max(precisions[ind]))
        else:
            max_precisions.append(0)
    print(max_precisions)
    average_precision = np.mean(np.array(max_precisions))

    return average_precision



def mean_average_precision(ground_truth_boxes, predicted_boxes):
    """ Calculates the mean average precision over the given dataset
        with IoU threshold of 0.5

    Args:
        ground_truth_boxes: (dict)
        {
            "img_id1": (np.array of float). Shape [number of GT boxes, 4]
        }
        predicted_boxes: (dict)
        {
            "img_id1": {
                "boxes": (np.array of float). Shape: [number of pred boxes, 4],
                "scores": (np.array of float). Shape: [number of pred boxes]
            }
        }
    """
    # DO NOT EDIT THIS CODE
    all_gt_boxes = []
    all_prediction_boxes = []
    confidence_scores = []

    for image_id in ground_truth_boxes.keys():
        pred_boxes = predicted_boxes[image_id]["boxes"]
        scores = predicted_boxes[image_id]["scores"]

        all_gt_boxes.append(ground_truth_boxes[image_id])
        all_prediction_boxes.append(pred_boxes)
        confidence_scores.append(scores)

    precisions, recalls = get_precision_recall_curve(
        all_prediction_boxes, all_gt_boxes, confidence_scores, 0.5)
    plot_precision_recall_curve(precisions, recalls)
    mean_average_precision = calculate_mean_average_precision(precisions, recalls)
    print("Mean average precision: {:.4f}".format(mean_average_precision))


if __name__ == "__main__":
    ground_truth_boxes = read_ground_truth_boxes()
    predicted_boxes = read_predicted_boxes()
    mean_average_precision(ground_truth_boxes, predicted_boxes)
