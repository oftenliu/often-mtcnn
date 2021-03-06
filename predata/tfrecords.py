import argparse
import os,sys
import tensorflow as tf
import numpy as np
import cv2
rootPath = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
sys.path.insert(0, rootPath)
from util import tfrecord_util

def __iter_all_data(net, iterType):
    saveFolder = os.path.join(rootPath, "tmp/data/%s/"%(net))
    if net not in ['pnet', 'rnet', 'onet']:
        raise Exception("The net type error!")
    if not os.path.isfile(os.path.join(saveFolder, 'pos.txt')):
        raise Exception("Please gen pos.txt in first!")
    if not os.path.isfile(os.path.join(saveFolder, 'landmark.txt')):
        raise Exception("Please gen landmark.txt in first!")
    if iterType == 'all':
        with open(os.path.join(saveFolder, 'pos.txt'), 'r') as f:
            pos = f.readlines()
        with open(os.path.join(saveFolder, 'neg.txt'), 'r') as f:
            neg = f.readlines()
        with open(os.path.join(saveFolder, 'part.txt'), 'r') as f:
            part = f.readlines()
        # keep sample ratio [neg, pos, part] = [3, 1, 1]
        base_num = min([len(neg), len(pos), len(part)])
        if len(neg) > base_num * 3:
            neg_keep = np.random.choice(len(neg), size=base_num * 3, replace=False)
        else:
            neg_keep = np.random.choice(len(neg), size=len(neg), replace=False)
        pos_keep = np.random.choice(len(pos), size=base_num, replace=False)
        part_keep = np.random.choice(len(part), size=base_num, replace=False)
        for i in pos_keep:
            yield pos[i]
        for i in neg_keep:
            yield neg[i]
        for i in part_keep:
            yield part[i]
        for item in open(os.path.join(saveFolder, 'landmark.txt'), 'r'):
            yield item
    elif iterType in ['pos', 'neg', 'part', 'landmark']:
        for line in open(os.path.join(saveFolder, '%s.txt'%(iterType))):
            yield line
    else:
        raise Exception("Unsupport iter type.")



def __get_dataset(net, iterType):
    dataset = []
    for line in __iter_all_data(net, iterType):
        info = line.strip().split(' ')
        data_example = dict()
        bbox = dict()
        data_example['filename'] = info[0]
        data_example['label'] = int(info[1])
        bbox['xmin'] = 0
        bbox['ymin'] = 0
        bbox['xmax'] = 0
        bbox['ymax'] = 0
        bbox['xlefteye'] = 0
        bbox['ylefteye'] = 0
        bbox['xrighteye'] = 0
        bbox['yrighteye'] = 0
        bbox['xnose'] = 0
        bbox['ynose'] = 0
        bbox['xleftmouth'] = 0
        bbox['yleftmouth'] = 0
        bbox['xrightmouth'] = 0
        bbox['yrightmouth'] = 0
        if len(info) == 6:
            bbox['xmin'] = float(info[2])
            bbox['ymin'] = float(info[3])
            bbox['xmax'] = float(info[4])
            bbox['ymax'] = float(info[5])
        if len(info) == 12:
            bbox['xlefteye'] = float(info[2])
            bbox['ylefteye'] = float(info[3])
            bbox['xrighteye'] = float(info[4])
            bbox['yrighteye'] = float(info[5])
            bbox['xnose'] = float(info[6])
            bbox['ynose'] = float(info[7])
            bbox['xleftmouth'] = float(info[8])
            bbox['yleftmouth'] = float(info[9])
            bbox['xrightmouth'] = float(info[10])
            bbox['yrightmouth'] = float(info[11])
        data_example['bbox'] = bbox
        dataset.append(data_example)
    return dataset



def _process_image_withoutcoder(filename):
    image = cv2.imread(filename)
    image_data = image.tostring()
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3
    return image_data, height, width


def _convert_to_example_simple(image_example, image_buffer):
    """
    covert to tfrecord file
    :param image_example: dict, an image example
    :param image_buffer: string, JPEG encoding of RGB image
    :param colorspace:
    :param channels:
    :param image_format:
    :return:
    Example proto
    """
    # filename = str(image_example['filename'])

    # class label for the whole image
    class_label = image_example['label']
    bbox = image_example['bbox']
    roi = [bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']]
    landmark = [bbox['xlefteye'], bbox['ylefteye'], bbox['xrighteye'], bbox['yrighteye'], bbox['xnose'], bbox['ynose'],
                bbox['xleftmouth'], bbox['yleftmouth'], bbox['xrightmouth'], bbox['yrightmouth']]

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': tfrecord_util._bytes_feature(image_buffer),
        'image/label': tfrecord_util._int64_feature(class_label),
        'image/roi': tfrecord_util._float_feature(roi),
        'image/landmark': tfrecord_util._float_feature(landmark)
    }))
    return example




def make_example(filename, image_example):
    """
    Loads data from image and annotations files and add them to a TFRecord.
    """
    image_data, height, width = _process_image_withoutcoder(filename)
    example = _convert_to_example_simple(image_example, image_data)
    return example

def gen_tfrecord(filename,net,iterType,shuffling):
    if tf.gfile.Exists(filename):
        tf.gfile.Remove(filename)
    # GET Dataset, and shuffling.
    dataset = __get_dataset(net=net, iterType=iterType)
    if shuffling:
        np.random.shuffle(dataset)
    # Process dataset files.
    # write the data to tfrecord
    tfrecord_writer = tf.python_io.TFRecordWriter(filename)

    for i, image_example in enumerate(dataset):
        #if i % 100 == 0:
            
        filename = image_example['filename']
        example = make_example(filename, image_example)
        tfrecord_writer.write(example.SerializeToString())
        print('\rConverting[%s]: %d/%d' % (net, i + 1, len(dataset)))

    tfrecord_writer.close()
    print('\n')

def start(net,shuffling=False):

    saveFolder = os.path.join(rootPath, "tmp/data/%s/"%(net))
    #tfrecord name
    if net == 'pnet':
        tfFileName = os.path.join(saveFolder, "all.tfrecord")
        gen_tfrecord(tfFileName, net, 'all', shuffling)
    elif net in ['rnet', 'onet']:
        #for n in ['pos', 'neg', 'part', 'landmark']:
        for n in ['landmark']:
            tfFileName = os.path.join(saveFolder, "%s.tfrecord"%(n))
            gen_tfrecord(tfFileName, net, n, shuffling)
    # Finally, write the labels file:
    print('\nFinished converting the MTCNN dataset!')







def parse_args():
    parser = argparse.ArgumentParser(description='Create hard bbox sample...',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--stage', dest='stage', help='working stage, can be pnet, rnet, onet',
                        default='unknow', type=str)
    parser.add_argument('--gpus', dest='gpus', help='specify gpu to run. eg: --gpus=0,1',
                        default='0', type=str)
    args = parser.parse_args()
    return args




if __name__ == "__main__":

    args = parse_args()
    stage = 'rnet'
    if stage not in ['pnet', 'rnet', 'onet']:
        raise Exception("Please specify stage by --stage=pnet or rnet or onet")
    # set GPU
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    start(stage, True)