import numpy as np
import os

"""
#　计算交并比
param box: 目标框　　左顶点坐标　　右下角坐标
param boxs_truth: 真是目标框集合

output:　交并比
"""

def IOU(bbox,bboxs_truth):

    bbox_area = (bbox[2] - bbox[0] + 1)*(bbox[3] - bbox[1]) #目标框面积

    boxs_truth_area = (bboxs_truth[:,2] - bboxs_truth[:,0] + 1) * (bboxs_truth[:,3] - bboxs_truth[:,1] +1)

    #交集区域
    intersection_topx = np.maximum(bbox[0],bboxs_truth[:,0])
    intersection_topy = np.maximum(bbox[1], bboxs_truth[:, 1])

    intersection_downx = np.minimum(bbox[2],bboxs_truth[:,2])
    intersection_downy = np.minimum(bbox[3],bboxs_truth[:,3])

    #求交集宽高

    intersection_width = np.maxinum(0, bboxs_truth[:,2] - bboxs_truth[:,0] + 1)  #保证宽高大于零
    intersection_height = np.maxinum(0, bboxs_truth[:, 3] - bboxs_truth[:, 1] + 1)

    intersection_area = intersection_width*intersection_height

    iou = intersection_area *1.0/(bbox_area + boxs_truth_area - intersection_area)
    return iou