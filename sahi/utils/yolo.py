import math 
import time

import cv2
import numpy as np

import torch
import torchvision.ops as ops
from torchvision import transforms as T

def box_iou(box1, box2):
    """Calculate Intersection over Union (IoU) for two sets of boxes.

    Args:
        box1 (Tensor[N, 4]): First set of boxes in (x1, y1, x2, y2) format.
        box2 (Tensor[M, 4]): Second set of boxes in (x1, y1, x2, y2) format.

    Returns:
        Tensor[N, M]: NxM matrix containing the pairwise IoU values for every 
        element in boxes1 and boxes2.
    """

    def box_area(box):
        """Calculate the area of the boxes.

        Args:
            box (Tensor): Box coordinates in (x1, y1, x2, y2) format.

        Returns:
            Tensor: Area of the boxes.
        """
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - 
             torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)

    return inter / (area1[:, None] + area2 - inter)  



def xywh2xyxy(x):
    """Convert bounding boxes from (center x, center y, width, height) to 
    (x1, y1, x2, y2) format.

    Args:
        x (Tensor or np.ndarray): nx4 boxes in [x, y, w, h] format.

    Returns:
        Tensor or np.ndarray: Converted boxes in [x1, y1, x2, y2] format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y



def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False, merge=False,
                        labels=()):
    """
    Perform Non-Maximum Suppression (NMS) on inference results.
    This function is based on https://github.com/WongKinYiu/yolov7

    Args:
        prediction (Tensor): Predictions from the model.
        conf_thres (float): Confidence threshold for filtering boxes.
        iou_thres (float): IoU threshold for NMS.
        classes (list, optional): Filter by class.
        agnostic (bool, optional): If True, class-agnostic NMS is applied.
        multi_label (bool, optional): If True, allows multiple labels per box.
        merge (bool, optional): If True, merges overlapping boxes.
        labels (tuple, optional): Ground truth labels for the image.

    Returns:
        List[Tensor]: List of detections, where each detection is a 
        (n, 6) tensor per image [xyxy, conf, cls].
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 500  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    # merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        if nc == 1:
            x[:, 5:] = x[:, 4:5] # for models with one class, cls_loss is 0 and cls_conf is always 0.5,
                                 # so there is no need to multiplicate.
        else:
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output



def yolo_postprocess(output, conf_thresh=0.25, iou_thresh=0.45, classes=None, agnostic=False, multi_label=False, merge=False, labels=()):
    """
    Post-process the YOLO model output.

    Args:
        output (torch.Tensor): The output tensor from the YOLO model.
        conf_thres (float): Confidence threshold for filtering boxes.
        iou_thres (float): IoU threshold for NMS.
        classes (list, optional): Filter by class.
        agnostic (bool, optional): If True, class-agnostic NMS is applied.
        multi_label (bool, optional): If True, allows multiple labels per box.
        merge (bool, optional): If True, merges overlapping boxes.
        labels (tuple, optional): Ground truth labels for the image.

    Returns:
        Returns:
        List[Tensor]: List of detections, where each detection is a 
        (n, 6) tensor per image [xyxy, conf, cls].
    """
    output = output[0]
    output = non_max_suppression(output, conf_thres=conf_thresh, iou_thres=iou_thresh, classes=classes, agnostic=agnostic, multi_label=multi_label, merge=merge, labels=labels)
    return output