import numpy as np
from PIL import Image
from skimage import io
from skimage.io import imread
import glob

from PIL import ImageFont
from PIL import ImageDraw

def save_image(array, i, label, folder, alphlist):
    array *= 255
    rescaled_im = np.array(array, dtype='uint8').reshape(20, 20)
    io.imsave("/home/havikbot/PycharmProjects/data_out/"+folder+"/"+str(i) + "-" + alphlist[label] + ".png", rescaled_im)

def convert_to_RGB(im):
    converted_im = np.zeros((im.shape[0], im.shape[1], 3), dtype='int8')
    for x in range(im.shape[0]):
        for y in range(im.shape[1]):
            for i in range(3):
                converted_im[x][y][i] = im[x][y]
    return converted_im


def set_red_point(image, x, y):
    try:
        image[y][x][0] = 255
        image[y][x][1] = 0
        image[y][x][2] = 0
    except:
        n = 2


def draw_line_and_letter(image, pos, size, letter):
    for x in range(pos[0], pos[0] + size[0]):
        set_red_point(image, x, pos[1])
        set_red_point(image, x, pos[1] + size[1] - 1)
    for y in range(pos[1], pos[1] + size[1]):
        set_red_point(image, pos[0], y)
        set_red_point(image, pos[0] + size[0] - 1, y)

    im = Image.fromarray(image, 'RGB')
    draw = ImageDraw.Draw(im)
    draw.text((pos[0] + (size[0] / 2) - 2, pos[1] - 5 + size[1] / 2), letter, (255, 0, 0))
    return np.array(im)


class detection():
    def __init__(self, folder):
        self.folder = folder
        self.alphlist = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                         'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']


    def detect_letters(self, model, save_images):  # Detection of letters

        prob_threshold = 0.95
        for img in glob.glob(self.folder + '/*.jpg'):
            detimg = imread(img, as_grey=True)
            output_image = imread(img, as_grey=True)
            output_image = convert_to_RGB(output_image)
            size_x = 20
            size_y = 20
            stepsize = 7
            test_images_raw = []
            print(detimg.shape)
            for y in range(0, detimg.shape[0], stepsize):
                for x in range(0, detimg.shape[1], stepsize):
                    np_im = np.array(detimg[y:(y + size_y), x:(x + size_x)], dtype="float")
                    np_im /= 255
                    padded = np.ones((20, 20), dtype="float")
                    padded[:np_im.shape[0], :np_im.shape[1]] = np_im
                    test_images_raw.append(padded.reshape(size_y, size_x, 1))

            count = 0
            for im in test_images_raw:
                # save_image(im, count, 0, "detection")
                count += 1

            test_images = np.array(test_images_raw)
            print(test_images.shape)

            predictions = model.predict(np.array(test_images))
            i = 0
            for y in range(0, detimg.shape[0], stepsize):
                for x in range(0, detimg.shape[1], stepsize):
                    pred = predictions[i]
                    if pred[np.argmax(pred)] >= prob_threshold:
                        label = self.alphlist[np.argmax(pred)]
                        print("letter at :", x, y, "with p=", pred[np.argmax(pred)])
                        if save_images: save_image(test_images_raw[i], i, np.argmax(pred), "detectionCorrect", self.alphlist)
                        output_image = draw_line_and_letter(output_image, [x, y], [size_x, size_y], label)
                    else:
                        # print("Not_letter at :", x, y, "with p=", pred[np.argmax(pred)])
                        if save_images: save_image(test_images_raw[i], i, 0, "detection2", self.alphlist)
                    i += 1

            img = Image.fromarray(output_image, 'RGB')
            img.show()



