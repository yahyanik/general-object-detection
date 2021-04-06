import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from object_detection.utils import visualization_utils as viz_utils
import cv2
import base64
from imutils.paths import list_images
import numpy as np
from PIL import Image
import io
import PIL
import imutils



class image_pre_load:
    st = ""

    def __init__(self):
        pass

    @staticmethod
    def load_from_java(image):
        decoded_data=base64.b64decode(image)
        np_data = np.fromstring(decoded_data, dtype=np.uint8)
        print(np_data.shape)
        img = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)
        return img

    @staticmethod
    def base64_to_pil(base64str):
        im_b64 = base64str.encode('utf-8')
        im_bytes = base64.b64decode(im_b64)  # im_bytes is a binary image
        im_file = io.BytesIO(im_bytes)  # convert image to file-like object
        img = Image.open(im_file)  # img is now PIL Image object
        return img

    @staticmethod
    def write(image1, txt, org, thickness, color):

        cv2.rectangle(image1, (50, 50), (65, 65), (0, 0,255), -1)
        image1 = cv2.putText(image1, txt, org, cv2.FONT_HERSHEY_SIMPLEX,
                            1, color, thickness, cv2.LINE_AA)
        return image1

    @staticmethod
    def rect(im, xs, ys, color, thickness):
        cv2.rectangle(im, xs, ys, color, thickness)
        return im

    @staticmethod
    def gray2rgb(im):
        return cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)


    @staticmethod
    def send_to_java(mask):
        pil_im = PIL.Image.fromarray(mask)
        buff = io.BytesIO()
        pil_im.save(buff, format="PNG")
        img_str = base64.b64encode(buff.getvalue())
        return img_str

    @staticmethod
    def imread(path):
        img = cv2.imread(path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @staticmethod
    def Camera_read(arg):
        vid = cv2.VideoCapture(arg)
        return vid

    @staticmethod
    def imread_from_folder(path):
        return list_images(path)

    @staticmethod
    def im_resize(im, image_size, imutils_resize=False):
        if not imutils_resize:
            return cv2.resize(im, image_size)
        else:
            return imutils.resize(im, width=image_size[0])

    @staticmethod
    def visualize(im):
        plt.figure()
        plt.imshow(im)
        return plt

    @staticmethod
    def show_visualize(plt):
        plt.show()

    @staticmethod
    def visualize_hist(hist):
        plt.figure()
        plt.bar([x for x in range(len(hist))], hist)
        plt.show()

    @staticmethod
    def negative_mask(mask, image):
        cp = image.cropped.copy()
        cp[cp != 0] = 255
        mask[mask == 0] = 1
        mask[mask == 255] = 0
        mask[mask == 1] = 255

        return cv2.bitwise_and(mask, cp[:,:,0])


    @staticmethod
    def plot(X, Y=None):
        plt.figure()
        if Y is not None:
            plt.plot(X, Y)
        else:
            plt.plot(X)
        return plt

    # def im_RGB(self, im):
    #     return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    #
    @staticmethod
    def merge(channels):
        return cv2.merge(channels)
    #
    @staticmethod
    def split( im):
        return cv2.split(im)
    #
    @staticmethod
    def YUV(im):
        return cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb)

    @staticmethod
    def erode(image):
        kernel = np.ones((3, 3), np.uint8)
        return cv2.erode(image, kernel, iterations=1)

    @staticmethod
    def YCR_CB2BGR (image):
        return cv2.cvtColor(image, cv2.COLOR_YCR_CB2BGR)
    #
    @staticmethod
    def gray(im):
        return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def brightest_channel(image):
        [R, G, B] = cv2.split(image)
        arg = np.argmax([np.mean(R), np.mean(G), np.mean(B)])
        gray = [R, G, B][arg]
        return gray

        return
    @staticmethod
    def im_YUV(im):
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)
        return gray

    @staticmethod
    def visualize_3d(data):
        x = np.arange(0, data.shape[1], 1)
        y = np.arange(0, data.shape[0], 1)
        X, Y = np.meshgrid(x, y)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        mycmap = plt.get_cmap('gist_earth')
        surf1 = ax.plot_surface(X, Y, data, cmap=mycmap)

        fig.colorbar(surf1, ax=ax, shrink=0.5, aspect=5)
        plt.show()

    @staticmethod
    def visualize_scatter(LIST_SCORES, three_d=False):
        fig = plt.figure()
        if three_d:
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter([i[0] for i in LIST_SCORES], [i[1] for i in LIST_SCORES], [i[2] for i in LIST_SCORES],
                   marker='^')  # bad

        plt.scatter([i[0] for i in LIST_SCORES], [i[1] for i in LIST_SCORES])
        plt.show()




def plot_detections(image_np,
                    boxes,
                    classes,
                    scores,
                    category_index,
                    figsize=(12, 16),
                    image_name=None):
    """Wrapper function to visualize detections.

    Args:
    image_np: uint8 numpy array with shape (img_height, img_width, 3)
    boxes: a numpy array of shape [N, 4]
    classes: a numpy array of shape [N]. Note that class indices are 1-based,
      and match the keys in the label map.
    scores: a numpy array of shape [N] or None.  If scores=None, then
      this function assumes that the boxes to be plotted are groundtruth
      boxes and plot all boxes as black with no classes or scores.
    category_index: a dict containing category dictionaries (each holding
      category index `id` and category name `name`) keyed by category indices.
    figsize: size for the figure.
    image_name: a name for the image file.
    """
    image_np_with_annotations = image_np.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
      image_np_with_annotations,
      boxes,
      classes,
      scores,
      category_index,
      use_normalized_coordinates=True,
      min_score_thresh=0.7)
    if image_name:
        plt.imsave(image_name, image_np_with_annotations)
    else:
        plt.imshow(image_np_with_annotations)