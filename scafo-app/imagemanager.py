import cv2
import matplotlib as mpl
import os
import yaml
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

# reading the config file
with open(r'/home/pi/scafo4.0/flask/scafo-app/config.yaml') as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    conf_file = yaml.load(file, Loader=yaml.FullLoader)

# path config
PATH = os.path.dirname(__file__)
output = PATH + conf_file['Flask']['path'] + '/output/'# MAN.output = 'static/output/'

class ImageManager:
    def __init__(self, output_path=output, display_scale=5, filename_prefix=''):
        self.output = output_path
        self.display_scale = display_scale
        self.filename_prefix = filename_prefix

    def get_output_path(self, filename, dir=''):
        if self.filename_prefix:
            filename = self.filename_prefix + '_' + filename
            if dir != '':
                filename = f'{dir}/{filename}'
        path = self.output + filename + '.png'
        return path

    def store_image(self, img, filename, dir=''):
        path = self.get_output_path(filename)
        # img0 = Image.fromarray(img)
        # img0.save(path)
        res = cv2.imwrite(path, img) # beacuse of color space
        
        if res:
            print(f'image saved: {path}')
            return path
        else:
            print(f'image failed to save: {path}')
            return False

    def get_image(self, filename, dir='',  grayscale=False):
        path = self.get_output_path(filename, dir)
        flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        img = cv2.imread(path, flag)
        return img

    def initialize_image_display(self, image):
        # resize to speed up the display
        img = cv2.resize(image, (image.shape[0]//self.display_scale, image.shape[1]//self.display_scale), interpolation=cv2.INTER_LINEAR)

        # convert color channels to int
        img = img.astype('uint8')

        # convert from grayscale image or bgr image to RGB
        if len(img.shape) == 3: 
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        return img

    def display_image(self, image, save_filename=''):
        if save_filename:
            self.store_image(image, save_filename)

        img = self.initialize_image_display(image)
        plt.figure(figsize = (12,8))
        plt.imshow(img, aspect='auto')
        plt.axis('off')
        plt.show()

    def display_images_compare(self, image1, image2, save_filename='', figsize=(8, 4)):
        if save_filename:
            filename1 = f'{save_filename}_1.png'
            self.store_image(image1, filename1)
            filename2 = f'{save_filename}_2.png'
            self.store_image(image2, filename2)

        image1 = self.initialize_image_display(image1)
        image2 = self.initialize_image_display(image2)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        ax1.imshow(image1, aspect='auto')
        ax1.axis('off')
        ax2.imshow(image2, aspect='auto')
        ax2.axis('off')
        plt.show()