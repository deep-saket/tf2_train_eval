###########################################################################################
#                                                                                         #
# This class contains code to evaluate object detection                                   #
#                                                                                         #
###########################################################################################
from cvml.metrics.KerasLikeMetrics import KerasLikeMetrics
from cvml.metrics.ObjectDetectionMetrics.BoundingBox import BoundingBox
from cvml.metrics.ObjectDetectionMetrics.BoundingBoxes import BoundingBoxes
from cvml.metrics.ObjectDetectionMetrics.Evaluator import *
from cvml.metrics.ObjectDetectionMetrics.utils import *

class PascalVOCObjectDetectionMetrics(KerasLikeMetrics):
    '''
    Computes precision, recall, AP, interpolated precision, interpolated recall
    for every class

    Args --
        bb_format -- str | one of ["xywh", "xyx2y2"]
        image_size -- tupple or list of length 2
        clss_ids -- tupple or list | a list of class ids in your annotation
                    e.g. [0, 1, 2, ..., n] in case of multiclass classification
                    e.g. [1] in case of binary classification
        coordinate_type -- str | one of ["absolute", "relative"]
        iou_threshold - float | default 0.3
    '''
    def __init__(self, bb_format, image_size, clss_ids, coordinate_type, iou_threshold=0.3) -> None:
        super(KerasLikeMetrics, self).__init__()
        
        # BoundingBox object to store all bounding boxes
        # Class representing bounding boxes (ground truths and detections)
        self.boundingboxes = BoundingBoxes()
        # number opf examples stored in order to be evaluated
        self.n = 0

        if bb_format not in ["xywh", "xyx2y2"]:
            print(f'[ERROR] | [PascalVOCObjectDetectionMetrics] | unsupported bb_format')
            exit()
        self.bb_format = BBFormat.XYWH if bb_format == "xywh" else BBFormat.XYX2Y2
        if len(image_size) != 2:
            print(f'[ERROR] | [PascalVOCObjectDetectionMetrics] | unsupported image_size')
            exit()
        self.image_size = image_size
        self.clss_ids = list(clss_ids)
        if coordinate_type not in ["absolute", "relative"]:
            print(f'[ERROR] | [PascalVOCObjectDetectionMetrics] | unsupported coordinate_type')
            exit()
        self.coordinate_type = CoordinatesType.Absolute if coordinate_type == "absolute" else CoordinatesType.Relative
        self.iou_threshold = iou_threshold
        
        # Create an evaluator object in order to obtain the metrics
        self.evaluator = Evaluator()

        
    def update_state(self, y_pred, y_true):
        '''
        Add example to metric evaluation list

        Args --
            y_pred -- list | predicted bboxes and classes | each elemet corresponds to 
                                        each example and each example has multiple predicted bboxs
                                        as ndarray; 0:4 contains bbox, 5 contains class confidence value, 
                                        6 contains class id
            y_true -- list | predicted bboxes and classes | each elemet corresponds to 
                                        each example and each example has multiple predicted bboxs
                                        as ndarray; 0:4 contains bbox, 5 contains class id
        '''
        if len(y_pred) != len(y_pred):
            print(f'[ERROR] | [PascalVOCObjectDetectionMetrics] | y_true and t_pred length missmatch')
            exit()

        for idx in range(len(y_pred)):
            # print()            
            ex_preds = y_pred[idx]
            ex_trues = y_true[idx]
            # print(idx, ex_preds)
            # print()
            # print(idx, ex_trues)
            
            for ex_true in ex_trues:
                idClass = ex_true[4]
                x = ex_true[0]
                y = ex_true[1]
                w = ex_true[2]
                h = ex_true[3]
                bb_true = BoundingBox(f'{self.n}', idClass, x, y, w, h,
                                      self.coordinate_type, self.image_size,
                                      BBType.GroundTruth, format=self.bb_format)
                self.boundingboxes.addBoundingBox(bb_true)
            for ex_pred in ex_preds:
                idClass = ex_pred[5]
                conf = ex_pred[4]
                x = ex_pred[0]
                y = ex_pred[1]
                w = ex_pred[2]
                h = ex_pred[3]
                bb_pred = BoundingBox(f'{self.n}', idClass, x, y, w, h,
                                      self.coordinate_type, self.image_size,
                                      BBType.Detected, conf, format=self.bb_format)
                self.boundingboxes.addBoundingBox(bb_pred)
            self.n += 1

    def result(self):
        '''
        Return computed metric result on the examples
        present in the evaluation list
        '''
        # Get metrics with PASCAL VOC metrics
        metricsPerClass = self.evaluator.GetPascalVOCMetrics(
            self.boundingboxes,  # Object containing all bounding boxes (ground truths and detections)
            IOUThreshold=self.iou_threshold,  # IOU threshold
            method=MethodAveragePrecision.EveryPointInterpolation)  # As the official matlab code
        print("Average precision values per class:\n")
        # Loop through classes to obtain their metrics
        average_precisions = []
        for mc in metricsPerClass:
            # Get metric values per each class
            c = mc['class']
            precision = mc['precision']
            recall = mc['recall']
            average_precision = mc['AP']
            ipre = mc['interpolated precision']
            irec = mc['interpolated recall']
            # Print AP per class
            print('%s: %f' % (c, average_precision))
            average_precisions.append(average_precision)

        mAP = sum(average_precisions) / len(average_precisions)
        return mAP

    def reset_state(self):
        '''
        Resets evaluation list to 0
        '''
        self.boundingboxes = BoundingBoxes()
        self.n = 0