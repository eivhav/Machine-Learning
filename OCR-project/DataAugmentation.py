
import numpy as np
import random

from scipy.ndimage import rotate, interpolation
from skimage.io import imread
from PIL import Image


class data_augmentation():
    def __init__(self):

        self.new_data_size = 10 # times the original size
        self.p_modified = 0.60
        self.p_noise = 0.40

        # if select to modification
        self.p_rotated = 0.25
        self.p_zoom_out = 0.20
        self.p_zoom_in = 0.20
        self.p_repositioned_and_overlapp = 0.35

        # if select for noise
        self.p_partial_image = 0.5
        self.p_constant_background = 0.35
        self.p_random_noise = 0.15


    def get_augmented_data(self, images, labels):
        images_with_labels = []
        for j in range(5):
            for i in range(len(images)):
                images_with_labels.append([images[i], labels[i]])

        modified = self.create_modified_images(images_with_labels, len(images_with_labels) * self.p_modified)
        noise = self.create_noised_images(images_with_labels, len(images_with_labels) * self.p_noise)

        print("modified: ", len(modified))
        print("noise: ", len(noise))

        all_data = images_with_labels + modified + noise
        print("all_data:" , len(all_data))
        random.shuffle(all_data)

        X = []
        y = []

        for i in range(len(all_data)):
            X.append(all_data[i][0])
            y.append(all_data[i][1])

        return X, y



    def create_modified_images(self, images_with_label, size):
        modified_images = []
        methods = np.random.choice(4, int(size),
                                  p=[self.p_rotated, self.p_zoom_out, self.p_zoom_in, self.p_repositioned_and_overlapp])

        for i in range(methods.shape[0]):
            im_label = random.choice(images_with_label)

            if methods[i] == 0:
                modified_images.append([self.rotate_image(im_label[0]), im_label[1]])
            elif methods[i] == 1:
                modified_images.append([self.zoom_im_out(im_label[0]), im_label[1]])
            elif methods[i] == 2:
                modified_images.append([self.zoom_im_in(im_label[0]), im_label[1]])
            elif methods[i] == 3:
                modified_images.append([self.reposition_im(im_label[0]), im_label[1]])

        return modified_images


    def create_noised_images(self, images_with_label, size):
        noise_images = []
        none_label = 0
        methods = np.random.choice(3, int(size),
                                   p=[self.p_partial_image, self.p_constant_background, self.p_random_noise])

        for i in range(methods.shape[0]):
            if methods[i] == 0:
                im1 = random.choice(images_with_label)[0]
                noise_images.append([self.create_partial_image(im1), none_label])

            elif methods[i] == 1:
                noise_images.append([self.create_constant_background(), none_label])
            elif methods[i] == 2:
                noise_images.append([self.create_random_noise(), none_label])

        return noise_images





    def rotate_image(self, im):
        # Figure out which way to rotate
        p = []
        for i in range(18): p.append(1-(i/18))
        total = sum(p)
        for i in range(18): p[i] = p[i] / total
        rotation = np.random.choice(18, 1, p=p) * 10
        if random.choice([False, True]): rotation = -rotation
        # Rotate image
        im = rotate(im, rotation, reshape=False)
        return im


    def zoom_im_out(self, im):
        size = random.choice([i for i in range(13, 16)])
        pos = [random.choice([i for i in range(0, 20 - size)]), random.choice([i for i in range(0, 20 - size)])]
        zoomed_im = interpolation.zoom(im, float(size) / 20, order=3, prefilter = True)
        out = np.ones((20, 20), dtype='float')
        out[pos[0]:(pos[0]+size), pos[1]:pos[1]+size] = zoomed_im
        # Maybe add both black and white backgrounds?
        return out


    def zoom_im_in(self, im):
        size = random.choice([i for i in range(13, 16)])
        pos = [random.choice([i for i in range(0, 20-size)]), random.choice([i for i in range(0, 20-size)]) ]
        cut_im = np.array([im[pos[0]:(pos[0]+size), pos[1]:pos[1]+size]])
        cut_im = cut_im.reshape(size, size)
        out = interpolation.zoom(cut_im, 20/size, order=3, prefilter = True)

        return out


    def reposition_im(self, im):
        prim_size = random.choice([i for i in range(2, 5)])
        sec_size = random.choice([i for i in range(0, 3)] + [0])

        width = False
        if random.choice([False, True]):
            dims = (20 + (prim_size*2), 20 + (2*sec_size))  # if width
            width = True
        else:
            dims = (20 + (sec_size*2), 20 + (2*prim_size))

        if random.choice([False, True]):  out = np.zeros(dims, dtype='float')  # Pad with zeroes or ones
        else: out = np.ones(dims, dtype='float')

        if width:
            out[prim_size:prim_size+20, sec_size:sec_size+20] = im[0:20, 0:20]
        else:
            out[sec_size:sec_size + 20, prim_size:prim_size + 20] = im[0:20, 0:20]

        corner = random.choice([i for i in range(0, 4)])
        if corner == 0:
            return out[0:20, 0:20]
        elif corner == 1:
            return out[out.shape[0]-20:, :20]
        elif corner == 2:
            return out[:20, out.shape[1] - 20:]
        elif corner == 3:
            return out[out.shape[0] - 20:, out.shape[1] - 20:]

        return im






    def create_partial_image(self, image1):
        if random.choice([False, True]):  # Pad with zeroes or ones
            temp = np.zeros((60, 60), dtype='float')
        else:
            temp = np.ones((60, 60), dtype='float')

        temp[20:40, 20:40] = image1[0:20, 0:20]

        out = np.zeros((20, 20), dtype='float')

        size1 = random.choice([i for i in range(6, 10)])
        size2 = random.choice([i for i in range(-5, 5)] + [0, 0])
        side = random.choice([i for i in range(0, 4)])

        if side == 0:
            out[0:20, 0:20] = temp[size1:size1+20, 20+size2:size2+40]
        if side == 1:
            out[0:20, 0:20] = temp[20+size2:size2+40, size1:size1+20]
        if side == 2:
            out[0:20, 0:20] = temp[40-size1:60-size1, 20+size2:size2+40]
        if side == 3:
            out[0:20, 0:20] = temp[20+size2:size2+40, 40-size1:60-size1]

        return out

    def create_constant_background(self):

        if random.choice([False, True]):  # Pad with zeroes or ones
            out = np.zeros((20, 20), dtype='float')
        else:
            out = np.ones((20, 20), dtype='float')

        color = float(random.randint(0,255)) / 255
        is_black_or_white = random.choice([i for i in range(0, 5)])
        if is_black_or_white == 0: color = 0.0
        elif is_black_or_white == 1: color = 1.0

        shade = np.ones((20, 20), dtype='float') * color
        width = random.randint(0,10)
        side = random.choice([i for i in range(0, 4)])

        if side == 0:
            out[0:width, 0:20] = shade[0:width, 0:20]
        if side == 1:
            out[0:20, 0:width] = shade[0:20, 0:width]
        if side == 2:
            out[20-width:20, 0:20] = shade[20-width:20, 0:20]
        if side == 3:
            out[0:20, 20-width:20] = shade[0:20, 20-width:20]

        return out


    def create_random_noise(self):

        out = np.zeros((20, 20), dtype='float')
        x_size = random.randint(1, 7)
        y_size = random.randint(1, 7)

        for x in range(int(20/x_size)):
            for y in range(int(20/y_size)):
                color = float(random.randint(0, 255)) / 255
                out[(x*x_size):(x*x_size)+x_size, (y*y_size):(y*y_size)+y_size] = color

            if 20 % y_size != 0:
                y = int(20/y_size)
                color = float(random.randint(0, 255)) / 255
                out[(x*x_size):(x*x_size)+x_size, (y*y_size):(y*y_size)+(20 % y_size)] = color

        if 20 % x_size != 0:
            x = int(20/x_size)
            for y in range(int(20/y_size)):
                color = float(random.randint(0, 255)) / 255
                out[(x * x_size):(x * x_size) + (x_size % 20), (y * y_size):(y * y_size) + y_size] = color

            if 20 % y_size != 0:
                y = int(20/y_size)
                color = float(random.randint(0, 255)) / 255
                out[(x * x_size):(x * x_size) + (x_size % 20), (y * y_size):(y * y_size) + (20 % y_size)] = color

        return out






def test_and_load_image(da):
    image_filepath = '/home/havikbot/PycharmProjects/testing/a_585.jpg'
    p_img = imread(image_filepath, as_grey=True)
    np_img = np.array(p_img) / 255

    # rotated_im = da.rotate_image(np_img) * 255
    for i in range(3):
        np_img = da.create_random_noise()

        print(np_img.shape)

        img = Image.fromarray(np_img*255)
        img.show()






