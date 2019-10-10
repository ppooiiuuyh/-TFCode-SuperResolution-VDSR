import sys
sys.path.append('../') #root path

import datetime
import time
import argparse
from utils.other_utils import *
import numpy as np

#tf.config.gpu.set_per_process_memory_fraction(0.4)
#tf.config.gpu.set_per_process_memory_growth(True)

""" --------------------------------------------------------------------
configuaration
---------------------------------------------------------------------"""
start = time.time()
time_now = datetime.datetime.now()
parser = argparse.ArgumentParser()
""" system """
parser.add_argument("--exp_type", type=int, default=0, help='experiment type')
parser.add_argument("--gpu", type=str, default=1)  # -1 for CPU
parser.add_argument("--model_tag", type=str, default="default", help='Exp name to save logs/checkpoints.')
parser.add_argument("--checkpoint_dir", type=str, default='../__outputs/checkpoints/', help='Dir for checkpoints.')
parser.add_argument("--summary_dir", type=str, default='../__outputs/summaries/', help='Dir for tensorboard logs.')
parser.add_argument("--restore_model_file", type=str, default=None, help='file for restoration')
#parser.add_argument("--restore_model_file", type=str, default='../__outputs/checkpoints/VDSR_VDSR_model_default_10_09_22_31_06/model.ckpt-30000', help='file for resotration')

""" model """
parser.add_argument("--batch_size", type=int, default=64, help='Minibatch size(global)')
parser.add_argument("--patch_size", type=int, default=41, help='Minipatch size(global)')
#parser.add_argument("--patch_stride", type=int, default=13, help='patch stride') #we just sample patches randomly for simplicity
parser.add_argument("--operating_channel", type=str, default="RGB", help="operating channel [RGB, YCBCR")  # -1 for CPU
parser.add_argument("--num_channels", type=int, default=3, help='the number of channels')
parser.add_argument("--scale", type=int, default=3, help='scaling factor')
parser.add_argument("--data_root_train", type=str, default="/projects/datasets/restoration/SR_training_datasets/T91", help='Data root dir')
parser.add_argument("--data_root_test", type=str, default="/projects/datasets/restoration/SR_testing_datasets/Set5", help='Data root dir')

""" training """
parser.add_argument("--learning_rate", type=float, default=0.0001, help="lr")
config = parser.parse_args()

""" tfconfig """
tfconfig = tf.ConfigProto()
tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.3

if config.exp_type == 0:
    from VDSR.utils.data_loader import *
    from models.model_VDSR import Model_Train
    config.model_tag = "VDSR_model_default"

def generate_expname_automatically():
    name = "VDSR_%s_%02d_%02d_%02d_%02d_%02d" % (config.model_tag,
            time_now.month, time_now.day, time_now.hour,
            time_now.minute, time_now.second)
    return name

expname  = generate_expname_automatically()
os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)


""" late directory checking """
config.checkpoint_dir += expname ; check_folder(config.checkpoint_dir)
config.summary_dir += expname ; check_folder(config.summary_dir)




""" --------------------------------------------------------------------
build model
---------------------------------------------------------------------"""
""" build model """
model = Model_Train(config, tfconfig)

""" restore model """
if config.restore_model_file is not None:
    model.restore_step()


""" --------------------------------------------------------------------
prepare dataset
---------------------------------------------------------------------"""
trainset_loader = Dataset_Loader(data_root_path = config.data_root_train, test_split_ratio = -1, config = config)
testset_loader = Dataset_Loader(data_root_path = config.data_root_test, test_split_ratio = -1, config = config) #-1 for using all



""" --------------------------------------------------------------------
train
---------------------------------------------------------------------"""
while True : #manuallry stopping
    """ train """
    log, output, label, input = model.train_step(trainset_loader, summarize_interval= 100)
    step = model.tensor2numpy(model.step)
    if step % 1 == 0:
        print("[train] step:{} elapse:{} {}".format(step, time.time() - start, log))
        result_concat = np.concatenate([input,output,label],axis=2)
        cv2.imshow('result_train_{}_{}'.format(config.exp_type,config.model_tag), resize(result_concat[...,::-1][0],1))
        cv2.waitKey(10)

    if step % 10 == 0:
        log_test, output_test, label_test, input_test = model.test_one_step(testset_loader.get_random_testset())
        print("[test one set] {}".format(log_test))
        result_concat = np.concatenate([input_test,output_test,label_test],axis=2)
        cv2.imshow('result_test_{}_{}'.format(config.exp_type,config.model_tag), resize(result_concat[...,::-1][0],1))
        cv2.waitKey(10)

    if step % 10 == 0:
        log_test = model.test_step(testset_loader.get_testset())
        print("[test] {}".format(log_test))

        #for e,o in  tqdm(enumerate(output)):
        #    cv2.imwrite(os.path.join(result_dir,"{}.jpg".format(e)), o[0][...,::-1])

    if step % 1000 == 0:
        save_path = model.save_step()
        print("[save] save path : {} ".format(save_path))

    model.increment_step(1)

