import numpy as np
import os

"""
#　计算交并比
param box: 目标框　　左顶点坐标　　右下角坐标
param boxs_truth: 真是目标框集合

output:　交并比
"""

def IOU(bbox,bboxs_truth):

    bbox_area = (bbox[2] - bbox[0] + 1)*(bbox[3] - bbox[1] + 1) #目标框面积

    boxs_truth_area = (bboxs_truth[:,2] - bboxs_truth[:,0] + 1) * (bboxs_truth[:,3] - bboxs_truth[:,1] +1)

    #交集区域
    intersection_topx = np.maximum(bbox[0],bboxs_truth[:,0])
    intersection_topy = np.maximum(bbox[1], bboxs_truth[:, 1])

    intersection_downx = np.minimum(bbox[2],bboxs_truth[:,2])
    intersection_downy = np.minimum(bbox[3],bboxs_truth[:,3])

    #求交集宽高

    intersection_width = np.maximum(0, intersection_downx - intersection_topx + 1)  #保证宽高大于零
    intersection_height = np.maximum(0, intersection_downy - intersection_topy + 1)

    intersection_area = intersection_width*intersection_height

    iou = intersection_area *1.0/(bbox_area + boxs_truth_area - intersection_area)
    return iou


def getBboxLandmarkFromTxt(txt, with_landmark=True):
    """
        Generate data from txt file
        return [(img_path, bbox, landmark)]
            bbox: [left, right, top, bottom]
            landmark: [(x1, y1), (x2, y2), ...]
    """
    dirname = os.path.dirname(txt)
    for line in open(txt, 'r'):
        line = line.strip()
        components = line.split(' ')
        img_path = os.path.join(dirname, components[0]) # file path
        # bounding box, (x1, y1, x2, y2)
        bbox = (components[1], components[3], components[2], components[4])
        bbox = [float(_) for _ in bbox]
        bbox = list(map(int, bbox))
        # landmark
        if not with_landmark:
            yield (img_path, BBox(bbox))
            continue
        landmark = np.zeros((5, 2))
        for index in range(0, 5):
            rv = (float(components[5+2*index]), float(components[5+2*index+1]))
            landmark[index] = rv
        #normalize
        '''
        for index, one in enumerate(landmark):
            rv = ((one[0]-bbox[0])/(bbox[2]-bbox[0]), (one[1]-bbox[1])/(bbox[3]-bbox[1]))
            landmark[index] = rv
        '''
        yield (img_path, BBox(bbox), landmark)

class BBox(object):
    """
        Bounding Box of face
    """

    def __init__(self, bbox):
        self.left = bbox[0]
        self.top = bbox[1]
        self.right = bbox[2]
        self.bottom = bbox[3]

        self.x = bbox[0]
        self.y = bbox[1]
        self.w = bbox[2] - bbox[0]
        self.h = bbox[3] - bbox[1]

    def expand(self, scale=0.05):
        bbox = [self.left, self.right, self.top, self.bottom]
        bbox[0] -= int(self.w * scale)
        bbox[1] += int(self.w * scale)
        bbox[2] -= int(self.h * scale)
        bbox[3] += int(self.h * scale)
        return BBox(bbox)

    # offset
    def project(self, point):
        x = (point[0] - self.x) / self.w
        y = (point[1] - self.y) / self.h
        return np.asarray([x, y])

    # absolute position(image (left,top))
    def reproject(self, point):
        x = self.x + self.w * point[0]
        y = self.y + self.h * point[1]
        return np.asarray([x, y])

    # landmark: 5*2
    def reprojectLandmark(self, landmark):
        p = np.zeros((len(landmark), 2))
        for i in range(len(landmark)):
            p[i] = self.reproject(landmark[i])
        return p

    # change to offset according to bbox
    def projectLandmark(self, landmark):
        p = np.zeros((len(landmark), 2))
        for i in range(len(landmark)):
            p[i] = self.project(landmark[i])
        return p

    # f_bbox = bbox.subBBox(-0.05, 1.05, -0.05, 1.05)
    # self.w bounding-box width
    # self.h bounding-box height
    def subBBox(self, leftR, rightR, topR, bottomR):
        leftDelta = self.w * leftR
        rightDelta = self.w * rightR
        topDelta = self.h * topR
        bottomDelta = self.h * bottomR
        left = self.left + leftDelta
        right = self.left + rightDelta
        top = self.top + topDelta
        bottom = self.top + bottomDelta
        return BBox([left, right, top, bottom])

