import os
import time

import numpy as np
import cv2
import random
from PIL import Image
import tensorflow as tf

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def normalize(images):
    return (images.astype(np.float32)/255.0)
def denormalize(images):
    return np.clip(images*255.0, a_min=0.001, a_max=254.99).astype(np.uint8)

def resize(image, scale = 0.25):
    size = (int(image.shape[0]*scale), int(image.shape[1]*scale))
    image = cv2.resize(image, (size[1],size[0]))
    return image

def expdim_2d_to_4d(img):
    [H,W,C] = np.shape(img)
    img = np.reshape(img, [1,H,W,C])
    return img

def img_fn_to_img(img_fn):
    img = cv2.imread(img_fn, cv2.IMREAD_COLOR)[..., ::-1]
    return img

def YCBCR2RGB(img):
    img = cv2.cvtColor(img[:, :, [0, 2, 1]], cv2.COLOR_YCR_CB2RGB)
    return img

def RGB2YCBCR(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)[:, :, [0, 2, 1]]
    return img

def random_crop(img,patch_size, seed = None):
    if seed is None:
        seed = time.time()
    random.seed(seed)
    H = img.shape[0]
    W = img.shape[1]
    hidx = random.randint(0,H-patch_size-1)
    widx = random.randint(0,W-patch_size-1)
    img_crop = img[hidx:hidx+patch_size, widx:widx+patch_size]

    return img_crop

def select_summary(predictions):
    def mkuint8(image):
        return tf.cast(image * 255, tf.uint8)

    summary_list = []
    for key, val in predictions.items():
        if "[i]" in key:   summary_list.append( tf.summary.image(key.replace("[i]",""), mkuint8(val), max_outputs=5))
        elif "[s]" in key: summary_list.append( tf.summary.scalar("losses/" + key.replace("[s]",""), val))

    summary_op = tf.summary.merge(summary_list)

    return summary_op

def restore_model(restore_model_file, sess, model_saver):
    model_saver.restore(sess, restore_model_file)



'''

def make_can_request(pos_correction_list):
    serial = ""
    print("num pos : ", len(pos_correction_list))
    def int2hex_ndigit(x,n=4):
        if x < 0:
            x = hex(((abs(x) ^ 0xfffff) + 1) & 0xfffff)
        else:
            x = hex(x)
            if len(x) < 2+n:
                x = x.replace("0x", "0x"+"0"*n)
        return x[-n:]


    for n in pos_correction_list:
        serial += (int2hex_ndigit(n,4))

    #checksum
    serial += int2hex_ndigit(0xfffff - sum(pos_correction_list) , 5)
    return serial

def resize(image, scale = 0.25):
    size = (int(image.shape[0]*scale), int(image.shape[1]*scale))
    image = cv2.resize(image, (size[1],size[0]))
    return image

def center_crop(image, size =(800,1200)):
    image = image[ image.shape[0]//2-size[0]//2 :image.shape[0]//2+size[0]//2    , image.shape[1]//2-size[1]//2 :image.shape[1]//2+size[1]//2]
    return image
'''