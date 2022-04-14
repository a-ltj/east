#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 19-4-8 下午10:24
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/CRNN_Tensorflow
# @File    : recongnize_chinese_pdf.py
# @IDE: PyCharm
"""
test the model to recognize the chinese pdf file
"""
import argparse
import os
import cv2
import numpy as np
import tensorflow as tf

from crnn.config import global_config
from crnn.crnn_model import crnn_net
from crnn.data_provider import tf_io_pipline_fast_tools

from east.icdar import get_images

CFG = global_config.cfg


def init_args():
    """

    :return: parsed arguments and (updated) config.cfg object
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str,
                        help='Path to the image to be tested',
                        default='data/test_images/test_01.jpg')
    parser.add_argument('--weights_path', type=str,
                        help='Path to the pre-trained weights to use')
    parser.add_argument('-c', '--char_dict_path', type=str,
                        help='Directory where character dictionaries for the dataset were stored')
    parser.add_argument('-o', '--ord_map_dict_path', type=str,
                        help='Directory where ord map dictionaries for the dataset were stored')
    # parser.add_argument('--save_path', type=str,
                        # help='The output path of recognition result')

    return parser.parse_args()

def get_images(image_path):
        '''
        find image files in test data path
        :return: list of files found
        '''
        files = []
        exts = ['jpg', 'png', 'jpeg', 'JPG']
        for parent, dirnames, filenames in os.walk(image_path):
            for filename in filenames:
                for ext in exts:
                    if filename.endswith(ext):
                        files.append(os.path.join(parent, filename))
                        break
        print('Find {} images'.format(len(files)))
        return files

def args_str2bool(arg_value):
    """

    :param arg_value:
    :return:
    """
    if arg_value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True

    elif arg_value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def split_pdf_image_into_row_image_block(pdf_image):
    """
    split the whole pdf image into row image block
    :param pdf_image: the whole color pdf image
    :return:
    """
    gray_image = cv2.cvtColor(pdf_image, cv2.COLOR_BGR2GRAY)
    binarized_image = cv2.adaptiveThreshold(
        src=gray_image,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=11,
        C=2
    )

    # sum along the row axis
    row_sum = np.sum(binarized_image, axis=1)
    idx_row_sum = np.argwhere(row_sum < row_sum.max())[:, 0]

    split_idx = []

    start_idx = idx_row_sum[0]
    for index, idx in enumerate(idx_row_sum[:-1]):
        if idx_row_sum[index + 1] - idx > 5:
            end_idx = idx
            split_idx.append((start_idx, end_idx))
            start_idx = idx_row_sum[index + 1]
    split_idx.append((start_idx, idx_row_sum[-1]))

    pdf_image_splits = []
    for index in range(len(split_idx)):
        idx = split_idx[index]
        pdf_image_split = pdf_image[idx[0]:idx[1], :, :]
        pdf_image_splits.append(pdf_image_split)

    return pdf_image_splits


def locate_text_area(pdf_image_row_block):
    """
    locate the text area of the image row block
    :param pdf_image_row_block: color pdf image block
    :return:
    """
    gray_image = cv2.cvtColor(pdf_image_row_block, cv2.COLOR_BGR2GRAY)
    binarized_image = cv2.adaptiveThreshold(
        src=gray_image,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=11,
        C=2
    )

    # sum along the col axis
    col_sum = np.sum(binarized_image, axis=0)
    idx_col_sum = np.argwhere(col_sum < col_sum.max())[:, 0]

    start_col = idx_col_sum[0] if idx_col_sum[0] > 0 else 0
    end_col = idx_col_sum[-1]
    end_col = end_col if end_col < pdf_image_row_block.shape[1] else pdf_image_row_block.shape[1] - 1

    return pdf_image_row_block[:, start_col:end_col, :]


def recognize(image_path,weights_path, char_dict_path, ord_map_dict_path):
    """

    :param image_path:
    :param weights_path:
    :param char_dict_path:
    :param ord_map_dict_path:
    :param output_path:
    :return:
    """
    # files = get_images(image_path)
    # pdf_recognize_results = []
    # for image_path in files:
        # print(image_path)
        # image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # # split pdf image into row block
        # pdf_image_row_blocks = split_pdf_image_into_row_image_block(image)

        # # locate the text area in each row block
        # pdf_image_text_areas = []
        # new_heigth = 32
        # max_text_area_length = -1
        # for index, row_block in enumerate(pdf_image_row_blocks):
            # text_area = locate_text_area(row_block)
            # text_area_height = text_area.shape[0]
            # scale = new_heigth / text_area_height
            # max_text_area_length = max(max_text_area_length, int(scale * text_area.shape[1]))
            # pdf_image_text_areas.append(text_area)
        # new_width = max_text_area_length
        # new_width = new_width if new_width > CFG.ARCH.INPUT_SIZE[0] else CFG.ARCH.INPUT_SIZE[0]
        
    #tf.reset_default_graph() 
        # definite the compute graph
    w=280
    h=32
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, dsize=tuple(CFG.ARCH.INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
            #img = cv2.resize(image,(280,32),)
           
    pdf_image_text_area = np.array(image, np.float32) / 127.5 - 1.0
    inputdata = tf.placeholder(
        dtype=tf.float32,
        shape=[1, h, w, CFG.ARCH.INPUT_CHANNELS],
        name='input'
    )

    codec = tf_io_pipline_fast_tools.CrnnFeatureReader(
        char_dict_path=char_dict_path,
        ord_map_dict_path=ord_map_dict_path
    )

    net = crnn_net.ShadowNet(
        phase='test',
        hidden_nums=CFG.ARCH.HIDDEN_UNITS,
        layers_nums=CFG.ARCH.HIDDEN_LAYERS,
        num_classes=CFG.ARCH.NUM_CLASSES
    )

    inference_ret = net.inference(
        inputdata=inputdata,
        name='shadow_net',
        reuse=False
    )

    decodes, _ = tf.nn.ctc_beam_search_decoder(
        inputs=inference_ret,
        sequence_length=int(w / 4) * np.ones(1),
        merge_repeated=False,
        beam_width=1
    )

        # config tf saver
    saver = tf.train.Saver()

        # config tf session
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TEST.TF_ALLOW_GROWTH

    sess = tf.Session(config=sess_config)

    with sess.as_default():
        ckpt_state = tf.train.get_checkpoint_state(weights_path)
        model_path = os.path.join(weights_path, os.path.basename(ckpt_state.model_checkpoint_path))
        #print('Restore from {}'.format(model_path))
        saver.restore(sess, model_path)
        #print(image_path)
        #files = get_images(image_path)
        pdf_recognize_results = []
        # for img_path in files:
            # #print(img_path)
            # image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            # image = cv2.resize(image, dsize=tuple(CFG.ARCH.INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
            # #img = cv2.resize(image,(280,32),)
           
            # pdf_image_text_area = np.array(image, np.float32) / 127.5 - 1.0
            #pdf_image_text_area = np.array(pdf_image_text_area, np.float32) / 127.5 - 1.0
        #preds = sess.run(decodes, feed_dict={inputdata: image})
        preds = sess.run(decodes, feed_dict={inputdata: [pdf_image_text_area]})
        preds = codec.sparse_tensor_to_str(preds[0])
        print('Predict image {:s} label {:s}'.format(os.path.split(image_path)[1], preds[0]))
        pdf_recognize_results.append(preds[0])
                    #单个保存
        #res_file = os.path.join(output_path,'{}.txt'.format('12_power')
        # res_file = output_path+'13_power.txt'
        # with open(res_file, 'w',encoding='utf_8') as f:
            # for x,y in zip(files,pdf_recognize_results):
                # i = os.path.split(x)[1]
                # print(i)
                # f.write('{},{}\n'.format(i, y))
                # image_name=os.path.split(img_path)[1]#12_power_img_43_box.jpg
                # (name,g)=os.path.splitext(image_name)#g=jpg;f=12_power_img_43_box
                # path_file_name = output_path+name+'.txt'
                # if not os.path.exists(path_file_name):
                    # with open(path_file_name, "w",encoding='utf-8') as file:
                        # res = image_name+","+'\n'.join(pdf_recognize_results)
                        # file.writelines(res)  
    # print(pdf_recognize_results)
    # print(type(pdf_recognize_results))
    return 
    #return res_file


if __name__ == '__main__':
    """

    """
    # init images
    args = init_args()

    # detect images
    recognize(
        #image_path=args.image_path,
        image_path = 'data/output/EAST/',
        #image_path=image_path,
        weights_path='model/checkpoint/crnnch/',
        char_dict_path='crnn/data/char_dict/char_dict_cn.json',
        ord_map_dict_path='crnn/data/char_dict/ord_map_cn.json',
        #output_path='data/outtxt/',
    )
