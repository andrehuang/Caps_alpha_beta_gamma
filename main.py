from __future__ import division
from keras.optimizers import RMSprop, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.layers import Input
from keras.models import Model
import os
import scipy.io as sio
import cv2
import sys
import numpy as np
from config import *
from models import sam_vgg, sam_resnet, schedule_vgg, schedule_resnet
from models import weighted_crossentropy, w_binary_crossentropy, w_categorical_crossentropy, kl_divergence
from scipy.misc import imread, imsave, imresize
import random
from parse_dataset import get_data
import data_augment
import bottleneck
from sklearn.metrics import recall_score
import keras_frcnn.resnet as resnet
from keras.utils.data_utils import get_file
from scipy.ndimage import filters
import matplotlib.pyplot as plt
from keras.losses import MSE

def generator(b_s, all_img_data, mode='train'):
    counter = 0
    sigma = 5
    while True:

        ims = np.zeros((b_s, img_size, img_size, 3))
        type_labels = np.zeros((b_s, type_num))

        part_labels = np.zeros((b_s, 28, 28, 8))
        part_body_labels = np.zeros((b_s, 28, 28, 2))
        full_body_labels = np.zeros((b_s, 28, 28, 1))

        for i in range(0, b_s):
            img_data_aug = all_img_data[(counter+i)%len(all_img_data)]
            try:
                # read in image, and optionally add augmentation
                # if mode == 'train':
                #     img_data_aug, x_img = data_augment.augment(img_data, augment=True)
                # else:
                #     img_data_aug, x_img = data_augment.augment(img_data, augment=False)
                x_img = cv2.imread(img_data_aug['filepath'])

                # w = int((img_data_aug['bboxes'][3] - img_data_aug['bboxes'][1]) / 4)
                # h = int((img_data_aug['bboxes'][2] - img_data_aug['bboxes'][0]) / 4)

                x = int((img_data_aug['bboxes'][3] + img_data_aug['bboxes'][1]) / 2)*img_size/x_img.shape[0]
                y = int((img_data_aug['bboxes'][2] + img_data_aug['bboxes'][0]) / 2)*img_size/x_img.shape[1]

                part_map = np.zeros((x_img.shape[0], x_img.shape[1], 8))# +0.01
                part_body_map = np.zeros((x_img.shape[0], x_img.shape[1], 2))
                full_body_map = np.zeros((x_img.shape[0], x_img.shape[1], 1))

                part_blur = np.zeros((28, 28, 8)) + 0.01
                part_body_blur = np.zeros((28, 28, 2)) + 0.01
                full_body_blur = np.zeros((28, 28, 1)) + 0.01

                # boxmap = np.zeros((x_img.shape[0], x_img.shape[1]))
                # boxmap[int(img_data_aug['bboxes'][1]):int(img_data_aug['bboxes'][3]),
                # int(img_data_aug['bboxes'][0]):int(img_data_aug['bboxes'][2])] = 1

                # full_body_map[x, y] = 1
                # full_body_map[:, :, 0] = filters.gaussian_filter(full_body_map[:, :, 0], [20, 20])

                smap = np.zeros((28, 28))
                smap[min(int(x/8),28), min(int(y/8),28)] = 1
                smap = filters.gaussian_filter(smap, [5, 5])
                smap[smap < 0] = 0

                # t = smap / smap.max()*255
                # imsave('full.png', t.astype(int))
                # imsave('img.png', x_img.astype(int))

                full_body_blur[:, :, 0] = smap / smap.max()
                full_body_labels[i, :] = np.expand_dims(np.copy(full_body_blur), axis=0)

                if img_data_aug['type_class'][0] == 1:
                    # a = max(img_data_aug['bboxes'][1], img_data_aug['land_map'][0])
                    # b = max(img_data_aug['bboxes'][0], img_data_aug['land_map'][1])
                    # c = min(img_data_aug['bboxes'][3], img_data_aug['land_map'][14])
                    # d = min(img_data_aug['bboxes'][2], img_data_aug['land_map'][15])
                    #
                    # part_body_map[int((img_data_aug['land_map'][0] + img_data_aug['land_map'][14]) / 2), int(
                    #     (img_data_aug['land_map'][1] + img_data_aug['land_map'][15]) / 2), 0] = 1
                    #
                    # part_body_map[:, :, 0] = filters.gaussian_filter(part_body_map[:, :, 0], [
                    #     int((img_data_aug['land_map'][14] - img_data_aug['land_map'][0]) / 2),
                    #     int((img_data_aug['land_map'][15] - img_data_aug['land_map'][1]) / 2)])
                    #
                    # smap = cv2.resize(part_body_map[:, :, 0], (28, 28), interpolation=cv2.INTER_CUBIC)
                    # smap[smap < 0] = 0
                    #
                    # t = smap / smap.max() * 255
                    # imsave('1.png', t.astype(int))

                    # part_body_blur[:, :, 0] = smap / smap.max()
                    part_body_blur[:, :, 0] = full_body_blur[:, :, 0]

                elif img_data_aug['type_class'][1] == 1:
                    # part_body_map[int((img_data_aug['land_map'][8] + img_data_aug['land_map'][14]) / 2), int(
                    #     (img_data_aug['land_map'][9] + img_data_aug['land_map'][15]) / 2), 1] = 1
                    #
                    # part_body_map[:, :, 1] = filters.gaussian_filter(part_body_map[:, :, 1], [
                    #     int((img_data_aug['land_map'][14] - img_data_aug['land_map'][8]) / 2),
                    #     int((img_data_aug['land_map'][15] - img_data_aug['land_map'][9]) / 2)])
                    #
                    # smap = cv2.resize(part_body_map[:, :, 1], (28, 28), interpolation=cv2.INTER_CUBIC)
                    # smap[smap < 0] = 0
                    #
                    # t = smap / smap.max() * 255
                    # imsave('1.png', t.astype(int))
                    #
                    # part_body_blur[:, :, 1] = smap / smap.max()
                    part_body_blur[:, :, 1] = full_body_blur[:, :, 0]

                else:
                    # part_body_map[int((img_data_aug['land_map'][0] + img_data_aug['land_map'][10]) / 2), int(
                    #     (img_data_aug['land_map'][1] + img_data_aug['land_map'][11]) / 2), 0] = 1
                    #
                    # part_body_map[:, :, 0] = filters.gaussian_filter(part_body_map[:, :, 0], [
                    #     int((img_data_aug['land_map'][10] - img_data_aug['land_map'][0]) / 2),
                    #     int((img_data_aug['land_map'][11] - img_data_aug['land_map'][1]) / 2)])
                    #
                    # smap = cv2.resize(part_body_map[:, :, 0], (28, 28), interpolation=cv2.INTER_CUBIC)
                    # smap[smap < 0] = 0
                    #
                    # t = smap / smap.max() * 255
                    # imsave('1.png', t.astype(int))
                    #
                    # part_body_blur[:, :, 0] = smap / smap.max()
                    #
                    # part_body_map[int((img_data_aug['land_map'][8] + img_data_aug['land_map'][14]) / 2), int(
                    #     (img_data_aug['land_map'][9] + img_data_aug['land_map'][15]) / 2), 1] = 1
                    #
                    # part_body_map[:, :, 1] = filters.gaussian_filter(part_body_map[:, :, 1], [
                    #     int((img_data_aug['land_map'][14] - img_data_aug['land_map'][8]) / 2),
                    #     int((img_data_aug['land_map'][15] - img_data_aug['land_map'][9]) / 2)])
                    #
                    # smap = cv2.resize(part_body_map[:, :, 1], (28, 28), interpolation=cv2.INTER_CUBIC)
                    # smap[smap < 0] = 0
                    #
                    # t = smap / smap.max() * 255
                    # imsave('1.png', t.astype(int))
                    #
                    # part_body_blur[:, :, 1] = smap / smap.max()

                    # part_body_map[int(img_data_aug['bboxes'][1] + (img_data_aug['bboxes'][3] - img_data_aug['bboxes'][1]) / 4), y, 0] = 1

                    # part_body_map[:, :, 0] = filters.gaussian_filter(part_body_map[:, :, 0], [10, 10])

                    # smap = cv2.resize(part_body_map[:, :, 0], (28, 28), interpolation=cv2.INTER_CUBIC)
                    smap = np.zeros((28, 28))
                    x1 = (img_data_aug['bboxes'][1] + (img_data_aug['bboxes'][3] - img_data_aug['bboxes'][1]) / 4)*img_size/x_img.shape[0]
                    smap[min(int(x1 / 8), 28), min(int(y / 8), 28)] = 1
                    smap = filters.gaussian_filter(smap, 3)
                    smap[smap < 0] = 0
                    # t = smap / smap.max() * 255
                    # imsave('up.png', t.astype(int))
                    part_body_blur[:, :, 0] = smap / smap.max()

                    # part_body_map[int(
                    #     img_data_aug['bboxes'][3] - (img_data_aug['bboxes'][3] - img_data_aug['bboxes'][1]) / 4), y, 1] = 1

                    # part_body_map[:, :, 1] = filters.gaussian_filter(part_body_map[:, :, 1], [10, 10])

                    # smap = cv2.resize(part_body_map[:, :, 1], (28, 28), interpolation=cv2.INTER_CUBIC)
                    smap = np.zeros((28, 28))
                    x2 = (img_data_aug['bboxes'][3] - (img_data_aug['bboxes'][3] - img_data_aug['bboxes'][1]) / 4)*img_size/x_img.shape[0]
                    smap[min(int(x2 / 8), 28), min(int(y / 8), 28)] = 1
                    smap = filters.gaussian_filter(smap, 3)
                    smap[smap < 0] = 0

                    # t = smap / smap.max() * 255
                    # imsave('low.png', t.astype(int))
                    part_body_blur[:, :, 1] = smap / smap.max()

                part_body_labels[i, :] = np.expand_dims(np.copy(part_body_blur), axis=0)

                for k in range(8):
                    if img_data_aug['land_vis'][k] == 1:
                        part_map[img_data_aug['land_map'][2*k], img_data_aug['land_map'][2*k+1], k] = 1
                        part_map[:, :, k] = filters.gaussian_filter(part_map[:, :, k], sigma)

                        smap = cv2.resize(part_map[:, :, k], (28, 28), interpolation=cv2.INTER_CUBIC)
                        smap[smap < 0] = 0
                        part_blur[:, :, k] = smap / smap.max()

                part_labels[i, :] = np.expand_dims(np.copy(part_blur), axis=0)

                # map = cv2.resize(map, (14, 14), interpolation=cv2.INTER_CUBIC)
                # plt.subplot(131)
                # plt.imshow(np.array(x_img))
                # plt.subplot(132)
                # plt.imshow(im_blur[:, :, 0])
                # plt.subplot(133)
                # plt.imshow(map[:, :, 0])
                # plt.show()

                # resize the image so that smalles side is length = 600px
                x_img = cv2.resize(x_img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)

                # try:
                #     y_rpn_cls, y_rpn_regr = calc_rpn(C, img_data_aug, width, height, img_size, img_size)
                # except:
                #     continue

                # Zero-center by mean pixel, and preprocess image
                # x_img = x_img[:, :, (2, 1, 0)]  # BGR -> RGB
                x_img = x_img.astype(np.float32)
                x_img[:, :, 0] -= img_channel_mean[0]
                x_img[:, :, 1] -= img_channel_mean[1]
                x_img[:, :, 2] -= img_channel_mean[2]
                x_img = x_img[:, :, ::-1]

                x_img = np.expand_dims(x_img, axis=0)

                ims[i, :] = np.copy(x_img)

            except Exception as e:
                print(e)
                continue

        yield ims, [part_labels, part_body_labels, full_body_labels, part_body_labels, full_body_labels,
                    part_body_labels, part_labels, part_labels, part_body_labels, full_body_labels]#
        counter = (counter + b_s) % len(all_img_data)


if __name__ == '__main__':
    if len(sys.argv) != 1:
        raise NotImplementedError
    else:
        seed = 7
        random.seed(seed)

        dataset_path = '/home/wwg/Documents/Dataset/deepfasion/FLD/'

        all_imgs = get_data(dataset_path)

        random.shuffle(all_imgs)
        num_imgs = len(all_imgs)

        train_imgs = [s for s in all_imgs if s['eva_status'] == 'train' or s['eva_status'] == 'val']
        val_imgs = [s for s in all_imgs if s['eva_status'] == 'test']
        # test_imgs  = [s for s in all_imgs if s['eva_status'] == 'test']

        print('Num train samples {}'.format(len(train_imgs)))
        print('Num val samples {}'.format(len(val_imgs)))
        # print('Num test samples {}'.format(len(test_imgs)))

        # phase = 'test'#sys.argv[1]
        phase = 'test'
        if phase == 'train':
            img_input = Input(batch_shape=(b_s, img_size, img_size, 3))
            # define the base network (resnet here, can be VGG, Inception, etc)
            #base_layers = resnet.nn_base(img_input, trainable=True)
            # share_layers = sam_resnet(base_layers)

            # x = Input(batch_shape=(b_s, img_size, img_size, 3))
            # x = Input((img_size, img_size, 3))
            # g = [1.]
            # p = [0.1]
            # import tensorflow as tf
            # p = tf.stack(p, name='ToFloat')
            # g = tf.stack(g, name='ToFloat')
            # k = weighted_crossentropy(g, p)
            # w = losses.binary_crossentropy(g, p)
            # sess = tf.Session()
            # print(sess.run(k))
            # print(sess.run(w))

            # fix random seed for reproducibility
            seed = 7
            np.random.seed(seed)

            if version == 0:
                m = Model(inputs=img_input, outputs=sam_vgg(img_input))

                # Load weights
                weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                        VGG_TF_WEIGHTS_PATH, cache_subdir='models')

                weights_path = 'weights.sam-vgg.05-196.3380.h5'

                m.load_weights(weights_path, by_name=True)

                print("Compili  ng SAM-VGG")
                m.compile(Adam(lr=1e-4), loss=[kl_divergence, kl_divergence, kl_divergence, kl_divergence,
                                               kl_divergence, kl_divergence, kl_divergence, kl_divergence,
                                               kl_divergence, kl_divergence])#
            elif version == 1:
                m = Model(inputs=img_input, outputs=sam_resnet(img_input))

                # Load weights
                weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                        RES_TF_WEIGHTS_PATH_NO_TOP, cache_subdir='models')
                print('loading weights from {}'.format(weights_path))
                m.load_weights(weights_path, by_name=True)
                # m.load_weights('weights.sam-resnet.06-15.6823.pkl')
                # m.load_weights('weights.sam-resnet.00-17.5592.pkl', by_name=True)
                print("Compiling SAM-ResNet")
                m.compile(Adam(lr=1e-4), loss=[kl_divergence, kl_divergence, kl_divergence])
                # m.compile(RMSprop(lr=1e-4), loss=weighted_crossentropy)
            else:
                raise NotImplementedError


            # tt = generator(b_s=b_s, all_img_data=train_imgs, mode='train')

            # if nb_imgs_train % b_s != 0 or nb_imgs_val % b_s != 0:
            #     print("The number of training and validation images should be a multiple of the batch size. "
            #           "Please change your batch size in config.py accordingly.")
            #     exit()
            if version == 0:
                print("Training SAM-VGG")
                m.fit_generator(generator(b_s=b_s, all_img_data=train_imgs, mode='train'), len(train_imgs)//b_s, epochs=nb_epoch,
                                validation_data=generator(b_s=b_s, all_img_data=val_imgs, mode='val'), validation_steps=len(val_imgs)//b_s,
                                callbacks=[EarlyStopping(patience=15),
                                           ModelCheckpoint('weights.sam-vgg.{epoch:02d}-{val_loss:.4f}.h5', save_best_only=False),
                                           LearningRateScheduler(schedule=schedule_vgg)])
            elif version == 1:
                print("Training SAM-ResNet")
                m.fit_generator(generator(b_s=b_s, all_img_data=train_imgs, mode='train'), len(train_imgs)//b_s, nb_epoch=nb_epoch,
                                validation_data=generator(b_s=b_s, all_img_data=val_imgs, mode='val'), nb_val_samples=len(val_imgs)//b_s,
                                callbacks=[EarlyStopping(patience=15),
                                           ModelCheckpoint('weights.sam-resnet.{epoch:02d}-{val_loss:.4f}.pkl', save_best_only=False),
                                           LearningRateScheduler(schedule=schedule_resnet)])
        elif phase == "test":

            img_input = Input(batch_shape=(1, img_size, img_size, 3))

            if version == 0:
                m = Model(inputs=img_input, outputs=sam_vgg(img_input))

            elif version == 1:
                m = Model(inputs=img_input, outputs=sam_resnet(img_input))

            else:
                raise NotImplementedError

            test_imgs = [s for s in all_imgs if s['eva_status'] == 'test']
            test_imgs = test_imgs[0:1000]
            nb_imgs_test = len(test_imgs)

            if version == 0:
                print("Loading SAM-VGG weights")
                m.load_weights('weights.sam-vgg.07-256.1460.h5')
            elif version == 1:
                print("Loading SAM-ResNet weights")
                m.load_weights('weights.sam-resnet.04-15.2714.pkl', by_name=True)
            print("Making prediction")
            nb_imgs_test = len(test_imgs)

            type_labels = np.zeros((nb_imgs_test, 3))
            gt_type_labels = np.zeros((nb_imgs_test, 3))

            land_dis = np.zeros((nb_imgs_test, 8))
            land_vis = np.zeros((nb_imgs_test, 8))

            part_body_dis = np.zeros((nb_imgs_test, 2))
            full_body_dis = np.zeros((nb_imgs_test))

            alpha_land_dis = np.zeros((nb_imgs_test, 8))
            alpha_part_body_dis = np.zeros((nb_imgs_test, 2))
            alpha_full_body_dis = np.zeros((nb_imgs_test))

            beta_part_body_dis = np.zeros((nb_imgs_test, 2))
            beta_full_body_dis = np.zeros((nb_imgs_test))

            gamma_land_dis = np.zeros((nb_imgs_test, 8))
            gamma_part_body_dis = np.zeros((nb_imgs_test, 2))

            alpha_land = np.zeros((nb_imgs_test, 28, 28, 8))
            alpha_part_body = np.zeros((nb_imgs_test, 28, 28, 2))
            alpha_full_body = np.zeros((nb_imgs_test, 28, 28, 1))

            beta_part_body = np.zeros((nb_imgs_test, 28, 28, 2))
            beta_full_body = np.zeros((nb_imgs_test, 28, 28, 1))

            gamma_land = np.zeros((nb_imgs_test, 28, 28, 8))
            gamma_part_body = np.zeros((nb_imgs_test, 28, 28, 2))

            com_land = np.zeros((nb_imgs_test, 28, 28, 8))
            com_part_body = np.zeros((nb_imgs_test, 28, 28, 2))
            com_full_body = np.zeros((nb_imgs_test, 28, 28, 1))

            for i in range(0, nb_imgs_test):
                img = cv2.imread(test_imgs[i]['filepath'])
                cv2.imwrite('fashionimages/'+str(i).zfill(5)+'.png', img)
                w = img.shape[0]
                h = img.shape[1]
                img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
                img = img.astype(np.float32)
                img[:, :, 0] -= img_channel_mean[0]
                img[:, :, 1] -= img_channel_mean[1]
                img[:, :, 2] -= img_channel_mean[2]
                img = img[:, :, ::-1]

                part_land_predictions, part_body_land_predictions, full_body_land_predictions, \
                beta_part_body_land_predictions, beta_full_body_land_predictions, \
                gamma_part_body_land_predictions, gamma_part_land_predictions, \
                com_part_land_predictions, com_part_body_land_predictions, com_full_body_land_predictions = \
                    m.predict(np.expand_dims(img, axis=0), batch_size=1)#

                alpha_land[i,:,:,:] = part_land_predictions
                alpha_part_body[i,:,:,:] = part_body_land_predictions
                alpha_full_body[i,:,:,:] = full_body_land_predictions

                beta_part_body[i,:,:,:] = beta_part_body_land_predictions
                beta_full_body[i,:,:,:] = beta_full_body_land_predictions

                gamma_land[i,:,:,:] = gamma_part_land_predictions
                gamma_part_body[i,:,:,:] = gamma_part_body_land_predictions

                com_land[i,:,:,:] = com_part_land_predictions
                com_part_body[i,:,:,:] = com_part_body_land_predictions
                com_full_body[i,:,:,:] = com_full_body_land_predictions
                # plt.subplot(191)
                # plt.imshow(img)
                # plt.subplot(192)
                # plt.imshow(part_land_predictions[0, :, :, 0])
                # plt.gray()
                # plt.subplot(193)
                # plt.imshow(part_land_predictions[0, :, :, 1])
                # plt.gray()
                # plt.subplot(194)
                # plt.imshow(part_land_predictions[0, :, :, 2])
                # plt.gray()
                # plt.subplot(195)
                # plt.imshow(part_land_predictions[0, :, :, 3])
                # plt.gray()
                # plt.subplot(196)
                # plt.imshow(part_land_predictions[0, :, :, 4])
                # plt.gray()
                # plt.subplot(197)
                # plt.imshow(part_land_predictions[0, :, :, 5])
                # plt.gray()
                # plt.subplot(198)
                # plt.imshow(part_land_predictions[0, :, :, 6])
                # plt.gray()
                # plt.subplot(199)
                # plt.imshow(part_land_predictions[0, :, :, 7])
                # plt.gray()
                # plt.show()

                # type_label = type_predictions[0, :]
                # gt_type_labels[i, :] = test_imgs[i]['type_class']
                # type_labels[i, bottleneck.argpartition(-type_label, 1)[:1]] = 1

                _m = int((test_imgs[i]['bboxes'][3] + test_imgs[i]['bboxes'][1]) / 2) * img_size / w
                _n = int((test_imgs[i]['bboxes'][2] + test_imgs[i]['bboxes'][0]) / 2) * img_size / h

                _positon = np.argmax(com_full_body_land_predictions[0, :, :, 0])
                x, y = divmod(_positon, 28)
                full_body_dis[i] = ((x*8-_m)**2+(y*8-_n)**2)**0.5/224

                _positon = np.argmax(full_body_land_predictions[0, :, :, 0])
                x, y = divmod(_positon, 28)
                alpha_full_body_dis[i] = ((x*8-_m)**2+(y*8-_n)**2)**0.5/224

                _positon = np.argmax(beta_full_body_land_predictions[0, :, :, 0])
                x, y = divmod(_positon, 28)
                beta_full_body_dis[i] = ((x * 8 - _m) ** 2 + (y * 8 - _n) ** 2) ** 0.5 / 224

                _m1 = (test_imgs[i]['bboxes'][1] + (
                    test_imgs[i]['bboxes'][3] - test_imgs[i]['bboxes'][1]) / 4) * img_size / w
                _m2 = (test_imgs[i]['bboxes'][3] - (
                    test_imgs[i]['bboxes'][3] - test_imgs[i]['bboxes'][1]) / 4) * img_size / w

                _positon = np.argmax(com_part_body_land_predictions[0, :, :, 0])
                x, y = divmod(_positon, 28)
                part_body_dis[i, 0] = ((x*8-_m1)**2+(y*8-_n)**2)**0.5/224

                _positon = np.argmax(com_part_body_land_predictions[0, :, :, 1])
                x, y = divmod(_positon, 28)
                part_body_dis[i, 1] = ((x * 8 - _m2) ** 2 + (y * 8 - _n) ** 2) ** 0.5 / 224

                _positon = np.argmax(part_body_land_predictions[0, :, :, 0])
                x, y = divmod(_positon, 28)
                alpha_part_body_dis[i, 0] = ((x * 8 - _m1) ** 2 + (y * 8 - _n) ** 2) ** 0.5 / 224

                _positon = np.argmax(part_body_land_predictions[0, :, :, 1])
                x, y = divmod(_positon, 28)
                alpha_part_body_dis[i, 1] = ((x * 8 - _m2) ** 2 + (y * 8 - _n) ** 2) ** 0.5 / 224

                _positon = np.argmax(beta_part_body_land_predictions[0, :, :, 0])
                x, y = divmod(_positon, 28)
                beta_part_body_dis[i, 0] = ((x * 8 - _m1) ** 2 + (y * 8 - _n) ** 2) ** 0.5 / 224

                _positon = np.argmax(beta_part_body_land_predictions[0, :, :, 1])
                x, y = divmod(_positon, 28)
                beta_part_body_dis[i, 1] = ((x * 8 - _m2) ** 2 + (y * 8 - _n) ** 2) ** 0.5 / 224

                _positon = np.argmax(gamma_part_body_land_predictions[0, :, :, 0])
                x, y = divmod(_positon, 28)
                gamma_part_body_dis[i, 0] = ((x * 8 - _m1) ** 2 + (y * 8 - _n) ** 2) ** 0.5 / 224

                _positon = np.argmax(gamma_part_body_land_predictions[0, :, :, 1])
                x, y = divmod(_positon, 28)
                gamma_part_body_dis[i, 1] = ((x * 8 - _m2) ** 2 + (y * 8 - _n) ** 2) ** 0.5 / 224

                for k in range(0, 8):
                    if test_imgs[i]['land_vis'][k] == 1:

                        _m = float(test_imgs[i]['land_map'][2*k])/w*224
                        _n = float(test_imgs[i]['land_map'][2*k+1])/h*224
                        land_vis[i, k] = 1

                        _positon = np.argmax(com_part_land_predictions[0, :, :, k])
                        x, y = divmod(_positon, 28)
                        land_dis[i, k] = ((x * 8 - _m) ** 2 + (y * 8 - _n) ** 2) ** 0.5 / 224

                        _positon = np.argmax(part_land_predictions[0, :, :, k])
                        x, y = divmod(_positon, 28)
                        alpha_land_dis[i, k] = ((x * 8 - _m) ** 2 + (y * 8 - _n) ** 2) ** 0.5 / 224

                        _positon = np.argmax(gamma_part_land_predictions[0, :, :, k])
                        x, y = divmod(_positon, 28)
                        gamma_land_dis[i, k] = ((x * 8 - _m) ** 2 + (y * 8 - _n) ** 2) ** 0.5 / 224



                        # plt.imshow(img)

            # type_recall = 0
            # type_total = 0
            #
            # for i in range(0, type_num):
            #     type_recall = type_recall+recall_score(gt_type_labels[:, i], type_labels[:, i])*sum(gt_type_labels[:, i])
            #     type_total = type_total + sum(gt_type_labels[:, i])
            #
            # print type_recall/type_total

            print sum(land_dis)/sum(land_vis), part_body_dis.mean(0), full_body_dis.mean()

            print sum(alpha_land_dis) / sum(land_vis), alpha_part_body_dis.mean(0), alpha_full_body_dis.mean()

            print beta_part_body_dis.mean(0), beta_full_body_dis.mean()

            print sum(gamma_land_dis) / sum(land_vis), gamma_part_body_dis.mean(0)

            sio.savemat('aresult.mat', {'alpha_land': alpha_land, 'alpha_part_body': alpha_part_body, 'alpha_full_body': alpha_full_body})
            sio.savemat('bresult.mat', {'beta_part_body': beta_part_body, 'beta_full_body': beta_full_body})
            sio.savemat('cresult.mat', {'gamma_land': gamma_land, 'gamma_part_body': gamma_part_body})
            sio.savemat('dresult.mat', {'com_land': com_land, 'com_part_body': com_part_body, 'com_full_body': com_full_body})


        else:
            raise NotImplementedError
