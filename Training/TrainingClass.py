import tensorflow
import h5py
import numpy as np
import os
import sys
import signal
import shutil
import importlib.util
import time
import math
import cv2


from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Model, Sequential, load_model
from keras.layers import * 
from keras import backend as K
from keras import losses
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger, TensorBoard, EarlyStopping
from keras.metrics import categorical_accuracy
from random import randint, shuffle, uniform
import imgaug as ia
from tensorflow.keras.utils import to_categorical
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa
from imgaug import parameters as iap
from sklearn.utils.extmath import cartesian

#def fancy_loss(y_true,y_pred):
#    "This function has been written in tensorflow, needs some little changes to work with keras"    
#    y_pred = tf.reshape(y_pred,[-1,y_pred.shape[-1]])
#    y_true = tf.argmax(y_true, axis=-1)
#    y_true = tf.reshape(y_true,[-1])
#    return z_softmax_flat(y_pred, y_true)


def shuffle_together_simple(images, labels, bodyparts):

    c = list(zip(images,labels, bodyparts))
    shuffle(c)
    images, labels, bodyparts = zip(*c)    
    images = np.asarray(images)
    labels = np.asarray(labels)
    bodyparts = np.asarray(bodyparts)
    
    return images, labels, bodyparts

def random_crop(x, y, permin, permax):
    h, w, _ = x.shape
    per_h = uniform(permin, permax)
    per_w = uniform(permin, permax)
    crop_size = (int((1-per_h)*h),int((1-per_w)*w))

    rangew = (w - crop_size[0]) // 2 if w>crop_size[0] else 0
    rangeh = (h - crop_size[1]) // 2 if h>crop_size[1] else 0
    offsetw = 0 if rangew == 0 else np.random.randint(rangew)
    offseth = 0 if rangeh == 0 else np.random.randint(rangeh)
    cropped_x = x[offseth:offseth+crop_size[0], offsetw:offsetw+crop_size[1], :]
    cropped_y = y[offseth:offseth+crop_size[0], offsetw:offsetw+crop_size[1], :]
    resize_x = cv2.resize(cropped_x, (h, w), interpolation=cv2.INTER_CUBIC)
    resize_y = cv2.resize(cropped_y, (h, w), interpolation=cv2.INTER_NEAREST)
    if cropped_y.shape[-1] == 0:
        return x, y
    else:
        return np.reshape(resize_x,(h,w,1)), resize_y


def dice_coef(y_true, y_pred):
    smooth=1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


### code fror Hausdorff disntance from https://github.com/danielenricocahall/Keras-Weighted-Hausdorff-Distance-Loss/blob/master/hausdorff/hausdorff.py
def cdist(A, B):
    """
    Computes the pairwise Euclidean distance matrix between two tensorflow matrices A & B, similiar to scikit-learn cdist.
    For example:
    A = [[1, 2],
         [3, 4]]
    B = [[1, 2],
         [3, 4]]
    should return:
        [[0, 2.82],
         [2.82, 0]]
    :param A: m_a x n matrix
    :param B: m_b x n matrix
    :return: euclidean distance matrix (m_a x m_b)
    """
    # squared norms of each row in A and B
    na = tf.reduce_sum(tf.square(A), 1)
    nb = tf.reduce_sum(tf.square(B), 1)

    # na as a row and nb as a co"lumn vectors
    na = tf.reshape(na, [-1, 1])
    nb = tf.reshape(nb, [1, -1])

    # return pairwise euclidead difference matrix
    D = tf.sqrt(tf.maximum(na - 2 * tf.matmul(A, B, False, True) + nb, 0.0))
    return D


def weighted_hausdorff_distance(w, h, alpha):
    all_img_locations = tf.convert_to_tensor(cartesian([np.arange(w), np.arange(h)]), dtype=tf.float32)
    max_dist = math.sqrt(w ** 2 + h ** 2)

    def hausdorff_loss(y_true, y_pred):
        def loss(y_true, y_pred):
            eps = 1e-6
            y_true = K.reshape(y_true, [w, h])
            gt_points = K.cast(tf.where(y_true > 0.5), dtype=tf.float32)
            num_gt_points = tf.shape(gt_points)[0]
            y_pred = K.flatten(y_pred)
            p = y_pred
            p_replicated = tf.squeeze(K.repeat(tf.expand_dims(p, axis=-1), num_gt_points))
            d_matrix = cdist(all_img_locations, gt_points)
            num_est_pts = tf.reduce_sum(p)
            term_1 = (1 / (num_est_pts + eps)) * K.sum(p * K.min(d_matrix, 1))

            d_div_p = K.min((d_matrix + eps) / (p_replicated ** alpha + (eps / max_dist)), 0)
            d_div_p = K.clip(d_div_p, 0, max_dist)
            term_2 = K.mean(d_div_p, axis=0)

            return term_1 + term_2

        batched_losses = tf.map_fn(lambda x:
                                   loss(x[0], x[1]),
                                   (y_true, y_pred),
                                   dtype=tf.float32)
        return K.mean(tf.stack(batched_losses))

    return hausdorff_loss


class TrainingClass:
    
    def __init__(self, name, model_path, data_path, save_folder, no_epochs, kernel_size, batch_size, filters, lrate = 1e-4, reg = 0.0001,  loss = 'categorical_crossentropy', duplicate = True ):
        self.name = name
        self.model_path = model_path
        self.data_path = data_path
        self.save_folder = save_folder
        self.kernel_size = kernel_size
        self.batch_size = batch_size
        self.filters = filters
        self.lrate = lrate
        self.reg = reg
        self.no_epochs = no_epochs
        if(loss == "dice"):
            self.loss = dice_coef_loss
        elif(loss == "hausdorff"):
            self.loss = weighted_hausdorff_distance
        else:
            self.loss = loss
        self.load_data()
        if(duplicate == True):
            self.train, self.train_label, self.train_bodypart = self.duplicate()
            print("Finished duplicating images, the final size of your training set is %d images."%self.train.shape[0])
        self.write_metadata()
        self.compile()
        
    def load_data(self):
        "Loads data from h5 file"
        hf = h5py.File(self.data_path, 'r')

        self.train = hf['train_img']
        self.no_images, self.height, self.width, self.channels= self.train.shape
        self.train_label = hf['train_label']
        self.train_bodypart = hf['train_bodypart'][:]
        self.no_images, _, _, self.no_classes = self.train_label.shape
        self.val = hf['val_img'][:]
        self.val_label = hf['val_label'][:]
        self.val_label = self.val_label.reshape((-1,self.height*self.width,self.no_classes))
        print("Data loaded succesfully.")
        
    def write_metadata(self):
        "Writes metadata to a txt file, with all the training information"
        metafile_path = self.save_folder + "/metadata.txt"

        if (os.path.isfile(metafile_path)):
            confirm_metada = input("Warning metadata file exists, continue? (y/n) ")
            if(confirm_metada == "y"):
                shutil.rmtree(metafile_path)
            else:
                sys.exit()
                
        metadata = open(metafile_path, "w")
        metadata.write("name: %s \n"%self.name)
        metadata.write("Data: %s \n"%self.data_path)
        metadata.write("kernel_size: %d \n" %self.kernel_size)
        metadata.write("batch_size:%d \n" %self.batch_size)
        metadata.write("filters %s \n" %(self.filters,))
        metadata.write("lrate: %f \n" %self.lrate)
        metadata.write("reg: %f \n" %self.reg)
        metadata.write("Loss function: %s \n" %self.loss)
        metadata.write("no_epochs: %d \n" %self.no_epochs)
        metadata.close()
                

        
    def generator(self):
        "This generator is used to feed the data to the training algorithm. Given a batch size, randomly divides the training data into batches. This function allows training even when all the data cannot be loaded into RAM memory."
        
        while True:
            indices = np.asarray(range(0, self.no_images))
            np.random.shuffle(indices)
            for idx in range(0, len(indices), self.batch_size):
                batch_indices = indices[idx:idx+self.batch_size]
                batch_indices.sort()
                batch_indices = batch_indices.tolist()
                by = self.train_label[batch_indices]
                by = by.reshape(-1, self.width*self.height, self.no_classes)
                bx = self.train[batch_indices]
                
                yield(bx,by)
    def duplicate(self):
        "Since our dataset is highly imbalanced among bodyparts, duplicate images from underrepresented bodyparts"
        img_per_category, counts = np.unique(self.train_bodypart, return_counts=True)
        img_per_category = dict(zip(img_per_category, counts))
        EXAMPLES_PER_CATEGORY = max(img_per_category.values())
        duplications_per_category = dict(img_per_category)
        for key in img_per_category:
            duplications_per_category[key] = int(EXAMPLES_PER_CATEGORY/img_per_category[key])

        duplicated_size = sum(duplications_per_category[k]*img_per_category[k] + img_per_category[k] \
                   for k in duplications_per_category)

        train_duplicated = np.zeros((duplicated_size,self.height,self.width,self.train.shape[3]))
        labels_duplicated = np.zeros((duplicated_size,self.height, self.width,self.no_classes))
        bodypart_duplicated = np.empty((duplicated_size),dtype = 'S10')

        train_duplicated[:self.no_images,...] = self.train
        labels_duplicated[:self.no_images,...] = self.train_label
        bodypart_duplicated[:self.no_images,...] = self.train_bodypart

        # Loop  over the different kind of bodyparts
        counter = self.no_images
        counter_block = 0
        for i, (k, v) in enumerate(duplications_per_category.items()):
            # Indices of images with a given bodypart
            indices = np.array(np.where(self.train_bodypart == k )[0])
            counter_block += len(indices)
            # Number of augmentation per image
            N = int(v)
            for j in indices:
                for l in range(N):
                    train_duplicated[counter,...] =self.train[j]
                    labels_duplicated[counter,...] = self.train_label[j]
                    bodypart_duplicated[counter] = k
                    counter +=1

        train_duplicated, labels_duplicated, bodypart_duplicated = shuffle_together_simple(train_duplicated, labels_duplicated, bodypart_duplicated)
        self.no_images = train_duplicated.shape[0]
        return train_duplicated, labels_duplicated, bodypart_duplicated

    def augmentator(self, index):
        " This function defines the trainsformations to apply on the images, and if required on the labels"

        translate_max = 0.01
        rotate_max = 15
        shear_max = 2

        affine_trasform = iaa.Affine( translate_percent={"x": (-translate_max, translate_max),
                                                         "y": (-translate_max, translate_max)}, # translate by +-
                                      rotate=(-rotate_max, rotate_max), # rotate by -rotate_max to +rotate_max degrees
                                      shear=(-shear_max, shear_max), # shear by -shear_max to +shear_max degrees
                                      order=[1], # use nearest neighbour or bilinear interpolation (fast)
                                      cval=125, # if mode is constant, use a cval between 0 and 255
                                      mode="reflect",
                                      #mode = "",
                                      name="Affine",
                                     )


        spatial_aug = iaa.Sequential([iaa.Fliplr(0.5), iaa.Flipud(0.5), affine_trasform])

        other_aug = iaa.SomeOf((1, None),
                [
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 0.4)), # blur images with a sigma between 0 and 1.0
                        iaa.ElasticTransformation(alpha=(0.5, 1.5), sigma=0.25), # very few

                    ]),

                ])

        '''
        affine_trasform = iaa.Affine( translate_percent={"x": (-translate_max, translate_max),
                                                         "y": (-translate_max, translate_max)}, # translate by +-
                                      rotate=(-rotate_max, rotate_max), # rotate by -rotate_max to +rotate_max degrees
                                      shear=(-shear_max, shear_max), # shear by -shear_max to +shear_max degrees
                                      order=[1], # use nearest neighbour or bilinear interpolation (fast)
                                      cval=125, # if mode is constant, use a cval between 0 and 255
                                      mode="reflect",
                                      name="Affine",
                                     )


        spatial_aug = iaa.Sequential([iaa.Fliplr(0.5), iaa.Flipud(0.5), affine_trasform])

        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        other_aug = iaa.SomeOf((1, None),
                [
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 0.4)), # blur images with a sigma between 0 and 1.0
                    ]),

                ])

        elastic_aug = iaa.SomeOf((1, None),
                [
                    iaa.OneOf([
                        sometimes(iaa.ElasticTransformation(alpha=(50, 60), sigma=16)), # move pixels locally around (with random strengths)
                    ]),

                ])

        
        # Defines augmentations to perform on the images and their labels
        augmentators = [spatial_aug,other_aug, elastic_aug]
        spatial_det = augmentators[0].to_deterministic()
        # to deterministic is needed to apply exactly the same spatial transformation to the data and the labels
        other_aug = augmentators[1]
        # When only adding noise there's no need to perform the transformation on the label
        elastic_det = augmentators[2].to_deterministic()
    
        image_aug = spatial_det.augment_image(self.train[index])
        label_aug = spatial_det.augment_image(255*self.train_label[index])

        image_aug = elastic_det.augment_image(image_aug)
        label_aug = elastic_det.augment_image(label_aug)
 
        img_crop, label_crop = random_crop(image_aug,label_aug,0.,0.4)
        image_aug = other_aug.augment_image(img_crop )    

        label_aug = label_crop
        
          
        label_aug = to_categorical(np.argmax(label_aug,axis=-1), num_classes = 3) # only needed if performing elastic transformations
        # Otherwise careful, returns [255,0,0] not [1,0,0] !
        '''
        augmentator = [spatial_aug,other_aug]
        spatial_det = augmentator[0].to_deterministic() 
        other_det = augmentator[1]

        image_aug = spatial_det.augment_image(self.train[index])
        label_aug = spatial_det.augment_image(self.train_label[index])
        img_crop, label_crop = random_crop(image_aug,label_aug,0.1,0.4)
        image_aug = other_det.augment_image(img_crop )               
        label_aug = to_categorical(np.argmax(label_crop,axis=-1), num_classes = self.no_classes)
        return image_aug, label_aug

    def generator_with_augmentations(self):
        "This generator is used to feed the data to the training algorithm. Given a batch size, randomly divides the training data into batches and augment each image once randomly. "
        batch_images = np.zeros((self.batch_size, self.width, self.height, 1))
        batch_labels = np.zeros((self.batch_size, self.width*self.height, self.no_classes))	# X and Y coordinates
        while True:
            indices = np.asarray(range(0, self.no_images))
            np.random.shuffle(indices)
            for idx in range(0, len(indices), self.batch_size):
                batch_indices = indices[idx:idx+self.batch_size]
                batch_indices.sort()
                batch_indices = batch_indices.tolist()
                for i, idx2 in enumerate(batch_indices):
                    augmented_image, augmented_label = self.augmentator(idx)
                    augmented_label = augmented_label.reshape(self.width*self.height, self.no_classes)
                    batch_images[i] = augmented_image
                    batch_labels[i] = augmented_label

                yield (batch_images,batch_labels)

    def compile(self):
        spec = importlib.util.spec_from_file_location("module.name", self.model_path)
        print(self.model_path)
        self.model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.model_module)
        self.model = self.model_module.model(l2_lambda = self.reg, input_shape = (self.height, self.width, self.channels), classes = self.no_classes, kernel_size = self.kernel_size, filter_depth = self.filters)
        self.model.compile(optimizer = rmsprop(lr = self.lrate, decay = 1e-6), loss = self.loss, metrics = ['accuracy'])
        #self.model.compile(optimizer = Adam(lr = self.lrate), loss = self.loss, metrics = ['accuracy'])
        #self.model.compile(optimizer = SGD(lr = self.lrate, momentum = 0.9, nesterov = True), loss = self.loss, metrics = ['accuracy'])
        print("Model loaded and compiled succesfully.")
        
    def fit(self):
        csv_logger = CSVLogger(self.save_folder + "/" + self.name + ".csv")
        #save_path = self.name + "_{epoch:03d}.h5"
        save_path = self.name + ".h5"
        save_path = self.save_folder + "/" + save_path
        earlystop = EarlyStopping(monitor="val_loss", min_delta = 0, patience = 20, verbose = 1, mode = 'min') 
        checkpoint = ModelCheckpoint(save_path, monitor = "val_loss", verbose = 1, save_best_only = True, save_weights_only = False, mode = "auto", period = 1)
        #tb = TensorBoard(log_dir = os.path.join(self.save_folder,'tboard'), batch_size = 1, write_graph = True, write_images = False)

        self.model.fit_generator(self.generator_with_augmentations(), steps_per_epoch = self.no_images // self.batch_size, epochs = self.no_epochs, callbacks = [csv_logger, checkpoint, earlystop], validation_data = (self.val, self.val_label))
        #self.model.fit_generator(self.generator(), steps_per_epoch = self.no_images // self.batch_size, epochs = self.no_epochs, callbacks = [csv_logger, checkpoint, earlystop], validation_data = (self.val, self.val_label))
    
        
        
