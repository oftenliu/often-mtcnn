#coding:utf-8
import sys
sys.path.append("..")
import argparse
import os
rootPath = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "./"))
sys.path.insert(0, rootPath)
from model.mtcnnmodel import mtcnn_pnet, mtcnn_rnet,mtcnn_onet

from util.loader import TestLoader
from detect.ronet_detect import ROnetDetect
from detect.pnet_detect import PnetDetect
from detect.detect import MtcnnDetector
import cv2

data_dir = './fddb'
out_dir = './fddb/Res'

def get_imdb_fddb(data_dir):
    imdb = []
    nfold = 10
    for n in range(nfold):
        file_name = 'FDDB-folds/FDDB-fold-%02d.txt' % (n + 1)
        file_name = os.path.join(data_dir, file_name)
        fid = open(file_name, 'r')
        image_names = []
        for im_name in fid.readlines():
            image_names.append(im_name.strip('\n'))      
        imdb.append(image_names)
    return imdb        



if __name__ == "__main__":

    stage = 'onet'
    detectors = [None, None, None]
    if stage in ['pnet', 'rnet', 'onet']:
        modelPath = os.path.join(rootPath, 'tmp/model/pnet/')
        a = [b[5:-6] for b in os.listdir(modelPath) if b.startswith('pnet-') and b.endswith('.index')]
        maxEpoch = max(map(int, a)) # auto match a max epoch model
        modelPath = os.path.join(modelPath, "pnet-%d"%(maxEpoch))
        print("Use PNet model: %s"%(modelPath))
        detectors[0] = PnetDetect(mtcnn_pnet,modelPath) 
    if stage in ['rnet', 'onet']:
        modelPath = os.path.join(rootPath, 'tmp/model/rnet/')
        a = [b[5:-6] for b in os.listdir(modelPath) if b.startswith('rnet-') and b.endswith('.index')]
        maxEpoch = max(map(int, a))
        modelPath = os.path.join(modelPath, "rnet-%d"%(maxEpoch))
        print("Use RNet model: %s"%(modelPath))
        detectors[1] = ROnetDetect(mtcnn_rnet, 24, 1, modelPath)
    if stage in ['onet']:
        modelPath = os.path.join(rootPath, 'tmp/model/onet/')
        a = [b[5:-6] for b in os.listdir(modelPath) if b.startswith('onet-') and b.endswith('.index')]
        maxEpoch = max(map(int, a))
        modelPath = os.path.join(modelPath, "onet-%d"%(maxEpoch))
        print("Use ONet model: %s"%(modelPath))
        detectors[2] = ROnetDetect(mtcnn_onet, 48, 1, modelPath)
    mtcnnDetector = MtcnnDetector(detectors=detectors, min_face_size = 24, threshold=[0.6, 0.5, 0.7])
    
    
    
    imdb = get_imdb_fddb(data_dir)
    nfold = len(imdb)    
    for i in range(nfold):
        image_names = imdb[i]
        print(image_names)
        dets_file_name = os.path.join(out_dir, 'fold-%02d-out.txt' % (i + 1))
        fid = open(dets_file_name,'w')
        sys.stdout.write('%s ' % (i + 1))
        image_names_abs = [os.path.join(data_dir,'originalPics',image_name+'.jpg') for image_name in image_names]
        test_data = TestLoader(image_names_abs)
        all_boxes, allLandmarks = mtcnnDetector.detect_face(test_data)
       
        for idx,im_name in enumerate(image_names):
            img_path = os.path.join(data_dir,'originalPics',im_name+'.jpg')
            image = cv2.imread(img_path)
            boxes = all_boxes[idx]
            if boxes is None:
                fid.write(im_name+'\n')
                fid.write(str(1) + '\n')
                fid.write('%f %f %f %f %f\n' % (0, 0, 0, 0, 0.99))
                continue
            fid.write(im_name+'\n')
            fid.write(str(len(boxes)) + '\n')
            
            for box in boxes:
                fid.write('%f %f %f %f %f\n' % (float(box[0]), float(box[1]), float(box[2]-box[0]+1), float(box[3]-box[1]+1),box[4]))                
                       
        fid.close()