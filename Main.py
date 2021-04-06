from image_read_preprocess import *
from model import *
import argparse
from eval import *


class main:


    model_obj = None
    train_obj = None
    eval_obj = None
    detections = None

    def __init__(self):
        self.args = self.args_method()
        self.read_image()
        # self.train()
        self.inference()
        self.eval()
        if self.args['show_gt_test']:
            self.show_gt_tests()

    def args_method(self):
        ap = argparse.ArgumentParser()

        ap.add_argument("-is", "--image_size", type=int,
                        default=640,
                        help="What would be the image size when training and inferencing")

        ap.add_argument("-td", "--train_image_dir", type=str,
                        default= 'models/research/object_detection/test_images/ducky/train/', # 'flower_test/train/', #'models/research/object_detection/test_images/ducky/train/'
                        help="training set image location")

        ap.add_argument("-tstd", "--test_image_dir", type=str,
                        default= 'models/research/object_detection/test_images/ducky/test/', # 'flower_test/test/',# 'models/research/object_detection/test_images/ducky/test/'
                        help="test set image location")

        ap.add_argument("-mcl", "--model_config", type=str,
                        default='models/research/object_detection/configs/tf2/ssd_resnet101_v1_fpn_640x640_coco17_tpu-8.config',
                        help="model config location")

        ap.add_argument("-mcpl", "--model_checkpoint", type=str,
                        default='models/research/object_detection/test_data/checkpoint/ckpt-0',
                        help="model checkpoint location")

        ap.add_argument("-tl", "--gt_boxes", type=str,
                        default=None,
                        help="training set annotation location")

        ap.add_argument("-sp", "--save_path", type=str,
                        default='saved_chckpoints/',
                        help="Where to save_trained model")

        ap.add_argument("-lp", "--load_path", type=str,
                        default='saved_chckpoints/',
                        help="Where to laod_trained model from")

        ap.add_argument("-on", "--object_name", type=str,
                        default= 'flowers', # 'flowers', # 'rubber_ducky',
                        help="what object this training is for?")

        ap.add_argument("-sts", "--show_training_samples", type=bool,
                        default=False,
                        help="saves training samples and their bounding on the working directory")

        ap.add_argument("-stts", "--show_test_samples", type=bool,
                        default=True,
                        help="saves test samples and their bounding on the working directory")

        ap.add_argument("-sgtt", "--show_gt_test", type=bool,
                        default=False,
                        help="show the ground-truth of the samples")

        ap.add_argument("-ua", "--use_augmentation", type=bool,
                        default=True,
                        help="use augmentation for training or not")

        ap.add_argument("-rl", "--read_labels", type=bool,
                        default=False, ######################### true for the flower
                        help="read labels from the hard disk")

        ap.add_argument("-lr", "--learning_rate", type=float,
                        default=0.01,
                        help="set learnong rate")

        ap.add_argument("-bs", "--batch_size", type=int,
                        default=4,
                        help="use augmentation for training or not")

        ap.add_argument("-itr", "--iteration", type=int,
                        default=500,
                        help="number of iterations")

        ap.add_argument("-ci", "--class_id", type=int,
                        default=1,
                        help="what is the id of this class")


        args = vars(ap.parse_args())
        return args

    def read_image(self):
        self.train_obj = training_data(args=self.args)
        self.train_obj.prepare_for_training()

    def train(self):
        self.model_obj = resnet101(args=self.args)
        self.model_obj.model = self.model_obj.load()
        # print(self.model_obj.model.summery())
        self.model_obj.training_loop(images_list=self.train_obj.train_images_np,
            gt_list = self.train_obj.gt_boxes,
            images_list_tensor = self.train_obj.train_image_tensors,
            gt_tensor = self.train_obj.gt_box_tensors,
            gt_one_hot_tensor = self.train_obj.gt_classes_one_hot_tensors,
            category_idx = self.train_obj.category_index)
        self.model_obj.save()

    def inference(self):
        if self.model_obj is None:
            self.model_obj = resnet101(args=self.args)
        self.detections = self.model_obj.inference()

    def eval(self):
        self.eval_obj = eval(self.args)
        if self.detections is None:
            self.eval_obj.load_model()

        #both the self.detections and data_dict are dics with the name of the image as key and the predictions and gt as values
            self.detections = self.eval_obj.inference()
        data_dict = self.eval_obj.get_gt()

        TP, FP, FN, AP = self.eval_obj.get_rates(self.detections, data_dict, path2test=self.args['test_image_dir'])

        print(f'this is the results for this class: {AP*100}')

    def show_gt_tests(self):
          # give boxes a score of 100%

        plt.figure()
        idx = 0
        class_id = self.args['class_id']
        category_index = {class_id: {'id': class_id, 'name': self.args['object_name']}}
        for imgPath in image_pre_load.imread_from_folder(self.args['test_image_dir']):
            img = cv2.imread(imgPath)
            # img = cv2.resize(img, (self.args['image_size'], self.args['image_size']))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.subplot(8, 7, idx + 1)
            idx+=1
            tree = ET.parse(imgPath[:-3]+'xml')
            root = tree._root
            num_boxes = FindNumberBoundingBoxes(root)
            file_name = [i.text for i in root.iter('filename')]
            recs = []
            h = img.shape[0]
            w = img.shape[1]
            for index in range(num_boxes):
                xmin = GetInt('xmin', root, index)/w
                ymin = GetInt('ymin', root, index)/h
                xmax = GetInt('xmax', root, index)/w
                ymax = GetInt('ymax', root, index)/h
                one_rec = [ymin, xmin, ymax, xmax]
                recs.append(one_rec)

            data_dict = np.array(recs, dtype=np.float32)
            dummy_scores = np.ones(shape=data_dict.shape[0], dtype=np.float32)
            plot_detections(
                img,
                data_dict,
                np.ones(shape=[data_dict.shape[0]], dtype=np.int32),
                dummy_scores, category_index, image_name=str(idx) + '.jpg')



if __name__ == '__main__':
    main()





