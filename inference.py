import sys
sys.path.append('../') #root path

import datetime
import time
import argparse
from utils.other_utils import *
import numpy as np

#tf.config.gpu.set_per_process_memory_fraction(0.6)
#tf.config.gpu.set_per_process_memory_growth(True)

""" --------------------------------------------------------------------
configuaration
---------------------------------------------------------------------"""
start = time.time()
time_now = datetime.datetime.now()
parser = argparse.ArgumentParser()
""" system """
parser.add_argument("--exp_type", type=int, default=0, help='experiment type')
parser.add_argument("--gpu", type=str, default=0)  # -1 for CPU
parser.add_argument("--restore_model_file", type=str, default="../__outputs/checkpoints/SRCNN_SRCNN_model_default_10_09_20_43_32/model.ckpt-2000", help='file for resotration')

""" model """
parser.add_argument("--operating_channel", type=str, default="RGB", help="operating channel [RGB, YCBCR")  # -1 for CPU
parser.add_argument("--num_channels", type=int, default=3, help='the number of channels')
parser.add_argument("--scale", type=int, default=3, help='scaling factor')
config = parser.parse_args()

if config.exp_type == 0:
    from SRCNN.utils.data_loader import *
    from models.inference_SRCNN import Model_Inference
    config.model_tag = "SRCNN_model_default"
os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)



""" --------------------------------------------------------------------
build model
---------------------------------------------------------------------"""
""" build model """
model = Model_Inference(config)

""" restore model """
if config.restore_model_file is not None:
    model.restore_step()


""" --------------------------------------------------------------------
inference
---------------------------------------------------------------------"""
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)
elapses = []
while True : #manuallry stopping
    """ capture """
    ret, img = cap.read()
    img_up = resize(img,scale = config.scale)
    img_up_tensor = np.expand_dims(img,-1)
    log, elapse, output, _ = model.inference(img_up_tensor)

    print(log)
    elapses.append(elapse)

    img_pad = np.pad(img, ((0,img_up.shape[0] - img.shape[0] ),(0,img_up.shape[1] - img.shape[1])))
    output_concat = np.concatenate([img_pad,img_up,output[0]],axis=2)
    cv2.imshow('result_test_{}_{}'.format(config.exp_type, config.model_tag), (output_concat[..., ::-1], 1))
    cv2.waitKey(10)



