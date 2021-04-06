from model import *
from image_read_preprocess import *
from utils import image_pre_load

class eval:
    def __init__(self, args, model=None):
        self.args = args
        self.model = model

    def load_model(self):
        if self.model is None:
            self.model_obj = resnet101(args=self.args)

    def inference(self):
        detections = self.model_obj.inference()
        return detections

    def get_gt(self):
        path = self.args['test_image_dir']
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

    def IOU(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        # return the intersection over union value
        return iou

    def get_rates(self, detections, data_dict, iou_tresh=0.5, acc=0.7, path2test='flower_test/test/'):

        TP = 0
        FP = 0
        FN = 0
        recall_image = []
        percision_image = []
        print(len(detections))
        print(len(data_dict))
        total_tp = 0
        for key in detections.keys():
            gt = data_dict[key]
            for idx in range(gt.shape[0]):
                total_tp += 1
        total_objects = total_tp

        total_tp = 0
        many_detected = 0
        current_recall = 0
        max_percision = 0
        for ii, key in enumerate(detections.keys()):
            print(f'{ii} is being processed')
            detection = detections[key]
            gt = data_dict[key]
            im = image_pre_load.imread(path2test+key)
            H = im.shape[0]
            W = im.shape[1]
            for idx in range(gt.shape[0]):
                gt_bbx = gt[idx, :]
                total_tp += 1
                # we are chacking the class, considering only having one class!
                gt_obj_detected_flag = False
                fn_flag = False

                if len(np.argwhere(detection['detection_scores'] > acc)) == 0:
                    continue
                many_detected += len(np.argwhere(detection['detection_scores'] > acc)[:,1])
                for j in np.argwhere(detection['detection_scores'] > acc)[:,1]:
                    pre = detection['detection_boxes'][0, j, :].numpy()
                    pre[0] = pre[0] * H
                    pre[1] = pre[1] * W
                    pre[2] = pre[2] * H
                    pre[3] = pre[3] * W

                    iou = self.IOU(gt_bbx, pre)
                    if iou > iou_tresh and not gt_obj_detected_flag:
                        TP += 1
                        gt_obj_detected_flag = True
                        if fn_flag:
                            FN -= 1
                    elif iou > iou_tresh and gt_obj_detected_flag:
                        TP -= 1
                        FP += 1
                    if iou < 0.05 and not gt_obj_detected_flag:
                        FN += 1
                        fn_flag = True
                if not gt_obj_detected_flag:
                    FP += 1

            #adding the max percision
            tmp_recall = TP / total_objects
            tmp_percision = TP / (many_detected if many_detected > 0 else 1)
            if tmp_recall == current_recall:
                if tmp_percision > max_percision:
                    max_percision = tmp_percision
            elif tmp_recall > current_recall:
                current_recall = tmp_recall
                max_percision = tmp_percision
            else:
                print('#############################################ERROR THIS IS NOT EXPECTED')


            recall_image.append(current_recall)
            percision_image.append(max_percision)

        sigma = 0
        for r in range(11):
            r = r/10
            for i, recall in enumerate(recall_image):
                if recall >= r:
                    sigma += percision_image[i]
                    break

        AP = (1/11) * sigma

        return TP, FP, FN, AP