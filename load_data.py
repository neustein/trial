import sys
import os
import numpy as np
import PIL.Image as Image
import pickle
import scipy.misc
import ipdb
import random

def load_data(path="./dataset/", mode='c'):
    train_img_list = []
    test_img_list = []
    train_label_list = []
    test_label_list = []

    label = 0
    label_name = []

    dir_imgs = os.listdir(path)
    print(path)
    print(dir_imgs)

    if mode == 'c':
        for dir_img in dir_imgs:
            print label
            print dir_img

            imgs = os.listdir(path+"/"+dir_img)
            random.shuffle(imgs)
            ret = make_image_set(path, dir_img, imgs, label)

            train_img_list += ret[0]
            train_label_list += ret[1]
            test_img_list += ret[2]
            test_label_list += ret[3] 

            label += 1
            label_name += [dir_img]

    elif mode == 'g':
        random.shuffle(dir_imgs)
        ret = make_image_set(dir_imgs, -1)
        train_img_list += ret[0]
        test_img_list += ret[2]

    else:
        print("mode takes 'c' or 'g'.")

    train_img_list = np.asarray(train_img_list).astype(np.float32)
    test_img_list = np.asarray(test_img_list).astype(np.float32)
    train_label_list = np.asarray(train_label_list).astype(np.int32)   
    test_label_list = np.asarray(test_label_list).astype(np.int32)   
    return [train_img_list, test_img_list],[train_label_list, test_label_list]


def make_image_set(path, dir_img, imgs, label=-1):
    train_img_list = []
    test_img_list = []
    train_label_list = []
    test_label_list = []
    if label != -1:
        num_test = int(len(imgs) * 0.2)
        num_train = len(imgs) - num_test
    else:
        num_train = len(imgs)

    count = 0
    for image in imgs:
        image_name, ext = os.path.splitext(image)
        if ext == ".png" or ext == ".jpg" or ext == ".jpeg" or ext == ".PNG" or ext == ".JPG":
            img = Image.open(path+"/"+dir_img+"/"+image).resize((32,32)).convert('RGB')
            if count < num_test:
                test_img_list += [np.asarray(img).transpose(2,0,1)]
                if label != -1:
                    test_label_list += [label]
            else:
                train_img_list += [np.asarray(img).transpose(2,0,1)]
                if label != -1:
                    train_label_list += [label]
        count += 1
                

    return [train_img_list, train_label_list, test_img_list, test_label_list]
