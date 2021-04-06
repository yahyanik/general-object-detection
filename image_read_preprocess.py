from utils import *
import tensorflow as tf
import numpy as np
import os
import xml.etree.ElementTree as ET




def GetItem(name, root, index=0):
    count = 0
    for item in root.iter(name):
        if count == index:
            return item.text
        count += 1
    # Failed to find "index" occurrence of item.
    return -1

def GetInt(name, root, index=0):
    return int(GetItem(name, root, index))

def FindNumberBoundingBoxes(root):
    index = 0
    while True:
        if GetInt('xmin', root, index) == -1:
            break
        index += 1
    return index


class training_data:
    def __init__(self, args=None):
        self.args = args
        self.category_index = None
        self.gt_classes_one_hot_tensors = None
        self.gt_box_tensors = None
        self.train_image_tensors = None
        self.train_images_np = None
        self.data_dict = None

        if self.args['gt_boxes']:
            self.gt_boxes = self.args['gt_boxes']
        elif self.args['read_labels']:
            self.gt_boxes = []
            self.data_dict = self.label()
        else:
            self.gt_boxes = [
                    np.array([[0.436, 0.591, 0.629, 0.712]], dtype=np.float32),
                    np.array([[0.539, 0.583, 0.73, 0.71]], dtype=np.float32),
                    np.array([[0.464, 0.414, 0.626, 0.548]], dtype=np.float32),
                    np.array([[0.313, 0.308, 0.648, 0.526]], dtype=np.float32),
                    np.array([[0.256, 0.444, 0.484, 0.629]], dtype=np.float32)
            ] * self.args['image_size']

            # self.gt_boxes = [
            #     np.array([[65, 113, 114, 179]], dtype=np.float32),
            #     np.array([[63, 86, 197, 195]], dtype=np.float32),
            #     np.array([[221, 747, 565, 1076]], dtype=np.float32),
            #     np.array([[69, 81, 108, 121]], dtype=np.float32),
            #     np.array([[356, 687, 528, 879]], dtype=np.float32),
            #     np.array([[24, 144, 487, 615]], dtype=np.float32),
            #     np.array([[91, 283, 214, 469]], dtype=np.float32),
            #     np.array([[286, 532, 503, 746]], dtype=np.float32),
            #     np.array([[122, 220, 463, 529]], dtype=np.float32),
            #     np.array([[71, 266, 137, 342]], dtype=np.float32),
            # ]
        self.read_images()

    def read_images(self):
        train_image_dir = self.args['train_image_dir']
        train_images_np = []
        flag = True if len(self.gt_boxes) == 0 else False
        for imagePath in image_pre_load.imread_from_folder(os.path.join(train_image_dir)):
            # image_path = os.path.join(train_image_dir, 'robertducky' + str(i) + '.jpg')
            img = image_pre_load.imread(imagePath)
            # img = cv2.resize(img, (self.args['image_size'], self.args['image_size']))
            if flag:
                self.gt_boxes.append(self.data_dict[os.path.basename(imagePath)])
            train_images_np.append(img)
        if self.args['object_name'] != 'rubber_ducky':
            for i, x in enumerate(train_images_np):
                arr = self.gt_boxes[i]
                H, W = x.shape[:2]
                arr[0,0] = arr[0,0] / H
                arr[0,2] = arr[0,2] / H
                arr[0,1] = arr[0,1] / W
                arr[0,3] = arr[0,3] / W
                self.gt_boxes[i] = arr
        for i, x in enumerate(train_images_np):
            train_images_np[i] = cv2.resize(x, (self.args['image_size'], self.args['image_size']))

        self.train_images_np = train_images_np

    def label(self):
        path = self.args['train_image_dir']
        l = os.listdir(path)
        data_dict = {}
        for file in l:
            if file[-3:] != 'xml':
                continue
            tree = ET.parse(os.path.join(path, file))
            root = tree._root
            num_boxes = FindNumberBoundingBoxes(root)
            file_name = [i.text for i in root.iter('filename')]
            recs = []
            for index in range(num_boxes):

                xmin = GetInt('xmin', root, index)
                ymin = GetInt('ymin', root, index)
                xmax = GetInt('xmax', root, index)
                ymax = GetInt('ymax', root, index)
                one_rec = [ymin, xmin, ymax, xmax]
                recs.append(one_rec)

            data_dict[file_name[0]] = np.array(recs, dtype=np.float32)
        return data_dict

    def show_training_samples(self):
        dummy_scores = np.array([1.0], dtype=np.float32)  # give boxes a score of 100%

        plt.figure()
        for idx in range(len(self.train_image_tensors)):
            plt.subplot(5, 4, idx + 1)
            # print(train_image_tensors[idx].numpy()[0,:,:,:].astype(np.uint8))
            print(f'for index {idx}: {self.gt_box_tensors[idx].numpy()}')
            plot_detections(
                self.train_image_tensors[idx].numpy()[0, :, :, :].astype(np.uint8),
                self.gt_box_tensors[idx].numpy(),
                np.ones(shape=[self.gt_box_tensors[idx].numpy().shape[0]], dtype=np.int32),
                dummy_scores, self.category_index, image_name=str(idx) + '.jpg')

            # img = self.train_image_tensors[idx].numpy()[0].astype(np.uint8)
            # H, W, _ = img.shape
            # print(int(self.gt_box_tensors[idx].numpy()[0][0]*W))
            # print(int(self.gt_box_tensors[idx].numpy()[0][1]*H))
            # print(int(self.gt_box_tensors[idx].numpy()[0][2]*W))
            # print(int(self.gt_box_tensors[idx].numpy()[0][3]*H))
            # print(H,W)
            # img = cv2.rectangle(img, (int(self.gt_box_tensors[idx].numpy()[0][0]*W), int(self.gt_box_tensors[idx].numpy()[0][1]*H)),
            #               (int(self.gt_box_tensors[idx].numpy()[0][2]*W), int(self.gt_box_tensors[idx].numpy()[0][3]*H)), (0, 255, 0), 2)
            # plt = image_pre_load.visualize(img)
            # image_pre_load.show_visualize(plt)
            # cv2.imshow('frame', img)
            # cv2.waitKey(0)
        # plt.show()

    def annotate_image_live(self):
        gt_boxes = []
        colab_utils.annotate(self.train_images_np, box_storage_pointer=gt_boxes)
        return gt_boxes

    def prepare_for_training(self):
        class_id = self.args['class_id']
        num_classes = 1

        self.category_index = {class_id: {'id': class_id, 'name': self.args['object_name']}}

        # Convert class labels to one-hot; convert everything to tensors.
        # The `label_id_offset` here shifts all classes by a certain number of indices;
        # we do this here so that the model receives one-hot labels where non-background
        # classes start counting at the zeroth index.  This is ordinarily just handled
        # automatically in our training binaries, but we need to reproduce it here.
        label_id_offset = 1
        train_image_tensors = []
        gt_classes_one_hot_tensors = []
        gt_box_tensors = []
        if self.args['use_augmentation']:
            for (train_image_np, gt_box_np) in zip(self.train_images_np, self.gt_boxes):

                image_tensor = tf.convert_to_tensor(train_image_np, dtype=tf.float32)
                train_image_tensors.append(tf.expand_dims(image_tensor, axis=0))
                flip_train_image_np = tf.image.flip_left_right(image_tensor)
                train_image_tensors.append(tf.expand_dims(flip_train_image_np, axis=0))

                gt_box_tensors.append(tf.convert_to_tensor(gt_box_np, dtype=tf.float32))
                dummy = gt_box_np[0][3]
                gt_box_np[0][3] = 1 - gt_box_np[0][1]
                gt_box_np[0][1] = 1 - dummy
                gt_box_tensors.append(tf.convert_to_tensor(gt_box_np, dtype=tf.float32))

                zero_indexed_groundtruth_classes = tf.convert_to_tensor(
                    np.ones(shape=[gt_box_np.shape[0]], dtype=np.int32) - label_id_offset)
                gt_classes_one_hot_tensors.append(tf.one_hot(
                    zero_indexed_groundtruth_classes, num_classes))
                gt_classes_one_hot_tensors.append(tf.one_hot(
                    zero_indexed_groundtruth_classes, num_classes))

        else:
            for (train_image_np, gt_box_np) in zip(self.train_images_np, self.gt_boxes):
                image_tensor = tf.convert_to_tensor(train_image_np, dtype=tf.float32)
                train_image_tensors.append(tf.expand_dims(image_tensor, axis=0))
                gt_box_tensors.append(tf.convert_to_tensor(gt_box_np, dtype=tf.float32))
                zero_indexed_groundtruth_classes = tf.convert_to_tensor(
                    np.ones(shape=[gt_box_np.shape[0]], dtype=np.int32) - label_id_offset)
                gt_classes_one_hot_tensors.append(tf.one_hot(
                    zero_indexed_groundtruth_classes, num_classes))

        print('Done prepping data.')

        self.gt_classes_one_hot_tensors = gt_classes_one_hot_tensors
        self.gt_box_tensors = gt_box_tensors
        self.train_image_tensors = train_image_tensors

        if self.args['show_training_samples']:
            self.show_training_samples()

