from utils import *

'''
1: 65 113, 114 179
10: 63 86, 197 195
1: 221 747, 565 1076
3: 69 81, 108 121
4: 356 687, 528 879
5: 24 144, 487 615
6: 91 283, 214 469
7: 286 532, 503 746
8: 122 220, 463 529
9: 71 266, 137 342
'''


train_path = 'flower_test/train'
for imagePath in image_pre_load.imread_from_folder(train_path):

    print(imagePath)
    image = image_pre_load.imread(imagePath)
    print(image.shape)
    plt = image_pre_load.visualize(image)
    image_pre_load.show_visualize(plt)
