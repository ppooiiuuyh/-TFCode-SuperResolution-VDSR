import sys
sys.path.append('../') #root path

import time
from VDSR.modules.networks import *
from VDSR.utils.data_loader import Dataset_Loader
import numpy as np
import tensorflow as tf
from functools import partial

class Model_Train():
    def __init__(self, config):
        self.config = config
        self.step = tf.Variable(0,dtype=tf.int64, name="global_step")
        tfconfig = tf.ConfigProto()
        tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.3
        self.sess = tf.Session(config = tfconfig)
        self.train_op, self.predictions_train, self.predictions_test = self.build_model()
        self.init_model()
        self.summary_writer, self.summary_train_op, self.summary_test_op = self.build_summarizer()
        self.model_saver = self.build_saver()

#======================================================
#              utils
#======================================================
    def tensor2numpy(self, tensor, feed= {}):
        return self.sess.run(tensor, feed_dict = feed)
    def increment_step(self, add):
        self.sess.run(self.step.assign_add(add))




#======================================================
#              model
#======================================================
    def build_model(self):
        "==========================  build model  ========================="
        train_op = None
        predictions_train = None
        predictions_test = None

        """ set placeholder """
        BS, H, W, CH = self.config.batch_size, self.config.patch_size, self.config.patch_size, self.config.num_channels
        self.input = tf.placeholder(tf.float32, shape=[BS, H, W, CH], name='input')
        self.label = tf.placeholder(tf.float32, shape=[BS, H, W, CH], name='label')

        self.input_test = tf.placeholder(tf.float32, shape=[1, None, None, CH], name='input_test')
        self.label_test = tf.placeholder(tf.float32, shape=[1, None, None, CH], name='label_test')

        """ model """
        G_Network = partial(VDSR_20, scope_name = "Generator")
        output = G_Network(self.input, num_channels = CH, is_training=True)
        output_test = G_Network(self.input_test, num_channels = CH, is_training=False)

        t_vars = tf.trainable_variables()
        print("------------------------------------------")
        print(" Trainable variables")
        print("------------------------------------------")
        for v in t_vars:
            print(v)
        print("------------------------------------------")

        """ loss function """
        loss = tf.reduce_mean(tf.square(output - self.label))

        """ optimizer """
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            G_op = tf.train.AdamOptimizer(self.config.learning_rate).minimize(loss, var_list=t_vars)
        #G_vars = [var for var in t_vars if "Generator" in var.name]

        """ output tensors """
        predictions_train = {
            "[i]inputs/input": self.input,
            "[i]outputs/generator": output,
            "[s]metric/PSNR": tf.reduce_mean(tf.image.psnr(self.label, output, max_val = 1)),
            "[s]metric/SSIM": tf.reduce_mean(tf.image.ssim(self.label, output, max_val = 1)),
            "[s]loss": loss,
        }
        predictions_test = {
            "[i]outputs/generator_test":output_test,
            "[_]metric/BICUBIC_PSNR_test": tf.reduce_mean(tf.image.psnr(self.label_test, self.input_test, max_val=1)),
            "[_]metric/PSNR_test": tf.reduce_mean(tf.image.psnr(self.label_test, output_test, max_val=1)),
            "[_]metric/SSIM_test": tf.reduce_mean(tf.image.ssim(self.label_test, output_test, max_val=1)),
        }
        train_op = {
            "G_op": G_op,
        }

        return train_op, predictions_train, predictions_test

    def init_model(self):
        init = tf.global_variables_initializer()
        self.sess.run(init)


    def build_summarizer(self):
        self.summaryholder_output = self.label_test
        self.summaryholder_PSNR = tf.placeholder(tf.float32)
        self.summaryholder_SSIM = tf.placeholder(tf.float32)
        summary_test = {
            "[i]outputs/generator_test": self.summaryholder_output,
            "[s]manual/PSNR_test": self.summaryholder_PSNR,
            "[s]manual/SSIM_test": self.summaryholder_SSIM,
        }

        summary_writer = tf.summary.FileWriter(self.config.summary_dir)
        summary_train_op = select_summary(self.predictions_train)
        summary_test_op = select_summary(summary_test)
        return summary_writer, summary_train_op, summary_test_op



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


    def train_step(self, iterator, summarize_interval = 100):
        input, label = iterator.__next__()
        self.sess.run(self.train_op["G_op"], feed_dict={self.input:input, self.label:label})
        predictions = self.sess.run(self.predictions_train, feed_dict={self.input:input, self.label:label})
        output = predictions["[i]outputs/generator"]

        """ return log str """
        reulst_log = "loss : {}, PSNR: {}, SSIM: {}".format(
            np.round_(predictions["[s]loss"], 4),
            np.round_(predictions["[s]metric/PSNR"],4),
            np.round_(predictions["[s]metric/SSIM"],4),
        )

        """ summarize """
        if self.tensor2numpy(self.step) % summarize_interval == 0:
            summary = self.sess.run(self.summary_train_op, feed_dict={self.input:input, self.label:label})
            self.summary_writer.add_summary(summary, self.tensor2numpy(self.step))
            self.summary_writer.flush()

        return reulst_log, denormalize(output), denormalize(label), denormalize(input)



    def test_one_step(self,oneset):
        elapses = []
        input_test, label_test = oneset
        start = time.time()
        predictions = self.sess.run(self.predictions_test, feed_dict={self.input_test:input_test, self.label_test:label_test})
        output = predictions["[i]outputs/generator_test"]
        elapse = time.time() - start
        elapses.append(elapse)

        """ return log str """
        log = "Median of elapse : {}, BICUBIC_PSNR: {}, PSNR: {}, SSIM: {}".format(
            np.round_(np.mean(elapses), 4),
            np.round_(predictions["[_]metric/BICUBIC_PSNR_test"], 4),
            np.round_(predictions["[_]metric/PSNR_test"], 4),
            np.round_(predictions["[_]metric/SSIM_test"], 4),
        )

        return log, denormalize(output), denormalize(label_test), denormalize(input_test)


    def test_step(self,iterator, do_summarize = True):
        elapses = []
        BICUBIC_PSRNs = []
        PSNRs = []
        SSIMs = []
        for input_test, label_test in iterator:
            start = time.time()
            predictions = self.sess.run(self.predictions_test, feed_dict={self.input_test:input_test, self.label_test:label_test})
            elapse = time.time() - start
            elapses.append(elapse)
            BICUBIC_PSRNs.append(predictions["[_]metric/BICUBIC_PSNR_test"])
            PSNRs.append(predictions["[_]metric/PSNR_test"])
            SSIMs.append(predictions["[_]metric/SSIM_test"])
        """ return log str """
        log = "Median of elapse : {}, BICUBIC_PSRN: {}, PSNR: {}, SSIM: {}".format(
            np.round_(np.mean(elapses), 4),
            np.round_(np.mean(BICUBIC_PSRNs), 4),
            np.round_(np.mean(PSNRs), 4),
            np.round_(np.mean(SSIMs), 4),
        )

        """ summarize """
        if do_summarize == True:
            output = predictions["[i]outputs/generator_test"] #summarize only last output
            summary = self.sess.run(self.summary_test_op,
                                    feed_dict={self.summaryholder_output:output,
                                               self.summaryholder_PSNR:np.round(np.mean(PSNRs), 4),
                                               self.summaryholder_SSIM:np.round(np.mean(SSIMs), 4)})
            self.summary_writer.add_summary(summary, self.tensor2numpy(self.step))
            self.summary_writer.flush()

        return log

""" ==================================================================================
                                module test
================================================================================== """
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    """ system """
    parser.add_argument("--exp_type", type=int, default=0, help='experiment type')
    parser.add_argument("--gpu", type=str, default=1)  # -1 for CPU
    parser.add_argument("--model_tag", type=str, default="default", help='Exp name to save logs/checkpoints.')
    parser.add_argument("--checkpoint_dir", type=str, default='../__outputs/checkpoints/', help='Dir for checkpoints.')
    parser.add_argument("--summary_dir", type=str, default='../__outputs/summaries/', help='Dir for tensorboard logs.')
    parser.add_argument("--restore_model_file", type=str, default=None, help='file for resotration')

    """ model """
    parser.add_argument("--batch_size", type=int, default=64, help='Minibatch size(global)')
    parser.add_argument("--patch_size", type=int, default=33, help='Minipatch size(global)')
    # parser.add_argument("--patch_stride", type=int, default=13, help='patch stride')
    parser.add_argument("--operating_channel", type=str, default="RGB", help="operating channel [RGB, YCBCR")  # -1 for CPU
    parser.add_argument("--num_channels", type=int, default=3, help='the number of channels')
    parser.add_argument("--scale", type=int, default=3, help='scaling factor')
    parser.add_argument("--data_root_train", type=str, default="/projects/datasets/restoration/SR_training_datasets/T91", help='Data root dir')
    parser.add_argument("--data_root_test", type=str, default="/projects/datasets/restoration/SR_testing_datasets/Set5", help='Data root dir')

    """ training """
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="lr")
    parser.add_argument("--learning_rate_sub", type=float, default=0.00001, help="lr2")
    config = parser.parse_args()


    """ check dataset loader """
    print("======= 1. check dataset loader ==========")
    trainset_loader = Dataset_Loader(data_root_path = config.data_root_train, test_split_ratio = -1, config = config)
    testset_loader = Dataset_Loader(data_root_path = config.data_root_test, test_split_ratio = -1, config = config)
    print("pass")
    print("========================================\n\n")


    """ check model """
    print("======= 2. check model ==========")
    model  = Model_Train(config)
    print("pass")
    print("========================================\n\n")


    """ check train step """
    print("======= 3. check train step ==========")
    model.train_step(trainset_loader, summarize_interval=100)
    print("pass")
    print("========================================\n\n")

    """ check test one step """
    print("======= 4. check test one step ==========")
    model.test_one_step(testset_loader.get_random_testset())
    print("pass")
    print("========================================\n\n")

    """ check test step """
    print("======= 5. check test step ==========")
    model.test_step(testset_loader.get_testset())
    print("pass")
    print("========================================\n\n")
