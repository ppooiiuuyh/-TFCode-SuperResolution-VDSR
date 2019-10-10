import sys
sys.path.append('../') #root path

import time
from SRCNN.modules.networks import *
import numpy as np
import tensorflow as tf
from functools import partial

class Model_Inference():
    def __init__(self, config):
        self.config = config
        self.sess = tf.Session()
        self.predictions_infer = self.build_model()
        self.init_model()
        self.model_saver = self.build_saver()

    #======================================================
    #              utils
    #======================================================
    def tensor2numpy(self, tensor, feed= {}):
        return self.sess.run(tensor, feed_dict = feed)


    #======================================================
    #              model
    #======================================================
    def build_model(self):
        "==========================  build model  ========================="
        predictions_infer = None

        """ set placeholder """
        CH = self.config.num_channels
        self.input_infer = tf.placeholder(tf.float32, shape=[1, None, None, CH], name='input_test')


        """ model """
        G_Network = partial(SRCNN_915, scope_name = "Generator")
        output_infer = G_Network(self.input_infer, num_channels = CH, is_training=False)

        t_vars = tf.trainable_variables()
        print("------------------------------------------")
        print(" Trainable variables")
        print("------------------------------------------")
        for v in t_vars:
            print(v)
        print("------------------------------------------")



        """ output tensors """
        predictions_infer = {
            "[i]inputs/input": self.input_infer,
            "[i]outputs/generator": output_infer,
        }

        return predictions_infer


    def init_model(self):
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def build_saver(self):
        model_saver = tf.train.Saver(max_to_keep=10)
        return model_saver


    #======================================================
    #              step
    #======================================================
    def save_step(self):
        save_path = self.model_saver.save(self.sess, os.path.join(self.config.checkpoint_dir, "model.ckpt"), global_step=self.step)
        return save_path
    def restore_step(self):
        self.model_saver.restore(self.sess, self.config.restore_model_file)



    def inference(self,image):
        start = time.time()
        predictions = self.sess.run(self.predictions_infer[""], feed_dict={self.input_test:image})
        output = predictions["[i]outputs/generator_test"]
        elapse = time.time() - start

        """ return log str """
        log = "Median of elapse : {}, ".format(
            np.round_(elapse, 4),
        )

        return log, elapse, denormalize(output), denormalize(image)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    """ system """
    parser.add_argument("--exp_type", type=int, default=0, help='experiment type')
    parser.add_argument("--gpu", type=str, default=0)  # -1 for CPU
    parser.add_argument("--restore_model_file", type=str, default="../../__outputs/checkpoints/SRCNN_SRCNN_model_default_10_09_20_43_32/model.ckpt-2000", help='file for resotration')

    """ model """
    parser.add_argument("--operating_channel", type=str, default="RGB", help="operating channel [RGB, YCBCR")  # -1 for CPU
    parser.add_argument("--num_channels", type=int, default=3, help='the number of channels')
    parser.add_argument("--scale", type=int, default=3, help='scaling factor')
    config = parser.parse_args()


    """ check model """
    print("======= 1. check model ==========")
    model  = Model_Train(config)
    model.restore_step()
    print("pass")
    print("========================================\n\n")
