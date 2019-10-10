import sys
sys.path.append('../')
import random
from tqdm import tqdm
from SRCNN.utils.other_utils import *
import time


class Dataset_Loader():
    def __init__(self,data_root_path, config, test_split_ratio=-1):
        self.data_root_path = data_root_path
        self.test_split_ratio = test_split_ratio
        self.config = config
        self.batch_size = config.batch_size
        self.patch_size = config.patch_size
        self.scale = config.scale
        self.operating_channel = config.operating_channel
        self.num_channels = config.num_channels
        _, self.labels_fn = self.load_dataset_fn()
        (_,self.labels_fn_train), (_, self.labels_fn_test) = self.split_fn(ratio=self.test_split_ratio)


    def load_dataset_fn(self):
        labels_fn = sorted([os.path.join(self.data_root_path, l) for l in os.listdir(self.data_root_path)])
        inputs_fn = None #it is not required in this task, but we leave it for generality


        print("========== load dataset file names===================")
        print("Find total {} label file names.".format(len(labels_fn)))
        print("ex) labels_fn: [{}]".format(labels_fn[0]))
        print("====================================================\n\n")

        return inputs_fn, labels_fn
    def split_fn(self, ratio=-1):
        if ratio == -1:
            inputs_fn_train = None
            inputs_fn_test = None
            labels_fn_train = self.labels_fn
            labels_fn_test = self.labels_fn
        else :
            num_inputs_test = None
            inputs_fn_train = None
            inputs_fn_test = None

            num_labels_test = int(len(self.labels_fn)*ratio)
            labels_fn_train = self.labels_fn[:-num_labels_test]
            labels_fn_test = self.labels_fn[num_labels_test:]

        return (inputs_fn_train, labels_fn_train), (inputs_fn_test, labels_fn_test)

    def bicubic_down_antialising_up(self, img, scale=3):
        # numpy
        img = denormalize(img)


        # PIL Image
        img_pil = Image.fromarray(img, "RGB")
        w, h = img_pil.size
        w_, h_ = w // scale, h // scale
        img_pil = img_pil.resize((w_, h_), Image.BICUBIC)
        #img_pil = img_pil.resize((w_, h_), Image.ANTIALIAS)
        img_pil = img_pil.resize((w, h), Image.BICUBIC)

        # numpy
        img = np.array(img_pil)
        img = normalize(img)

        return img
    def fn_to_single_image_pair(self, img_fn):
        img = img_fn_to_img(img_fn)
        img = normalize(img)

        label = img
        input = self.bicubic_down_antialising_up(img, scale = self.scale)

        if self.operating_channel == "YCBCR":
            label = RGB2YCBCR(label)
            input = RGB2YCBCR(input)

        return input, label


    def __iter__(self):
        return self
    def __next__(self):
        inputs = []
        labels = []

        while len(inputs) < self.batch_size:
            #pick random
            rand_idx = random.randint(0,len(self.labels_fn_train)-1)
            input, label = self.fn_to_single_image_pair(self.labels_fn_train[rand_idx])

            #preprocess
            seed = time.time()
            input = random_crop(input, patch_size=self.patch_size, seed = seed)
            label = random_crop(label, patch_size=self.patch_size, seed = seed)
            input = input[...,0:self.num_channels]
            label = label[...,0:self.num_channels]

            #append
            inputs.append(input)
            labels.append(label)

        return np.array(inputs), np.array(labels)

    def get_random_testset(self):
        # pick
        rand_idx = random.randint(0, len(self.labels_fn_test) - 1)
        input, label = self.fn_to_single_image_pair(self.labels_fn_test[rand_idx])

        # preprocess
        input = input[..., 0:self.num_channels]
        label = label[..., 0:self.num_channels]
        input = expdim_2d_to_4d(input)
        label = expdim_2d_to_4d(label)
        return input, label


    def get_testset(self):
        inputs = []
        labels = []

        for i in range(len(self.labels_fn_test)) :
            #pick
            input, label = self.fn_to_single_image_pair(self.labels_fn_test[i])

            # preprocess
            input = input[..., 0:self.num_channels]
            label = label[..., 0:self.num_channels]
            input = expdim_2d_to_4d(input)
            label = expdim_2d_to_4d(label)

            #append
            inputs.append(input)
            labels.append(label)

        return zip(inputs, labels)



""" ==================================================================================
                                module test
================================================================================== """
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=2, help='Minibatch size(global)')
    parser.add_argument("--patch_size", type=int, default=33, help='Minipatch size(global)')
    parser.add_argument("--operating_channel", type=str, default="RGB", help="operating channel [RGB, YCBCR")  # -1 for CPU
    parser.add_argument("--num_channels", type=int, default=3, help='the number of channels')
    parser.add_argument("--scale", type=int, default=3, help='scaling factor')
    parser.add_argument("--data_root_train", type=str, default="/projects/datasets/restoration/SR_training_datasets/T91", help='Data root dir')
    parser.add_argument("--data_root_test", type=str, default="/projects/datasets/restoration/SR_testing_datasets/Set5", help='Data root dir')
    config = parser.parse_args()

    data_loader = Dataset_Loader(data_root_path = config.data_root_train, config = config)


    for i in range(10):
        input, label = data_loader.get_random_testset()
        print(input.shape, label.shape)
        cv2.imshow("image",np.concatenate([input,label],axis=1)[...,::-1][0])
        cv2.waitKey(0)

    for input,label in data_loader.get_testset():
        print(input.shape, label.shape)
        cv2.imshow("image",np.concatenate([input,label],axis=1)[...,::-1][0])
        cv2.waitKey(0)

    for input,label in data_loader:
        print(input.shape, label.shape)
        cv2.imshow("image",np.concatenate([input,label],axis=1)[...,::-1][0])
        cv2.waitKey(0)





