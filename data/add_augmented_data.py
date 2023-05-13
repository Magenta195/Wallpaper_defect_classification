from typing import List
import os
import random
from optparse import OptionParser, OptionGroup

import numpy as np
import PIL
from PIL import Image


def save_file(image:PIL.Image, path:str, title:str):
    dir_path = os.path.dirname(path)
    file_name = path.split('/')[-1]
    image.save(os.path.join(dir_path, title+'_'+file_name))


def random_crop(path_list:List[str], crop_rate:float=0.7):
    """This function replicates the random crop process"""
    for path in path_list:
        image = np.asarray(Image.open(path))
        img_size = image.shape
        # extracting channels
        channel_0, channel_1, channel_2 = image[:,:,0], image[:,:,1], image[:,:,2]

        # deriving random idx
        crop_size = (int(img_size[0] * crop_rate), int(img_size[1] * crop_rate))

        idx_row = random.randint(0, img_size[0] - crop_size[0])
        idx_column = random.randint(0, img_size[1] - crop_size[1])

        # cropping
        channel_0 = channel_0[idx_row: idx_row + crop_size[0],
                              idx_column: idx_column + crop_size[1]]
        channel_1 = channel_1[idx_row: idx_row + crop_size[0],
                              idx_column: idx_column + crop_size[1]]
        channel_2 = channel_2[idx_row: idx_row + crop_size[0],
                              idx_column: idx_column + crop_size[1]]

        image = np.dstack((channel_0, channel_1, channel_2))
        image = Image.fromarray(image)
        save_file(image, path, 'cropped')    


def noise_image(path_list:List[str], noise_intensity:float=0.2):
    """This function replicates the image noising process"""
    noise_threshold = 1 - noise_intensity

    for path in path_list:
        image = np.asarray(Image.open(path))
        img_size = image.shape
        flatten_size = img_size[0] * img_size[1]

        # extracting channels
        channel_0, channel_1, channel_2 = image[:,:,0], image[:,:,1], image[:,:,2]

        #  flatenning channels
        channel_0 = channel_0.reshape(flatten_size)
        channel_1 = channel_1.reshape(flatten_size)
        channel_2 = channel_2.reshape(flatten_size)

        #  creating vector of zeros
        noise_0 = np.zeros(flatten_size, dtype='uint8')
        noise_1 = np.zeros(flatten_size, dtype='uint8')
        noise_2 = np.zeros(flatten_size, dtype='uint8')

        #  noise probability
        for idx in range(flatten_size):
            regulator = round(random.random(), 1)
            if regulator > noise_threshold:
                noise_0[idx] = 255
                noise_1[idx] = 255
                noise_2[idx] = 255
            elif regulator == noise_threshold:
                noise_0[idx] = 0
                noise_1[idx] = 0
                noise_2[idx] = 0
            else:
                noise_0[idx] = channel_0[idx]
                noise_1[idx] = channel_1[idx]
                noise_2[idx] = channel_2[idx]
        
        #  reshaping noise vectors
        noise_0 = noise_0.reshape(img_size[:-1])
        noise_1 = noise_1.reshape(img_size[:-1])
        noise_2 = noise_2.reshape(img_size[:-1])

        #  stacking images
        image = np.dstack((noise_0, noise_1, noise_2))
        image = Image.fromarray(image)
        save_file(image, path, 'noise')


def select_dir(path:str, num_files:int):
    """Select directories from limited conditions"""
    if not os.path.isdir(path):
        raise ValueError('Invalid Path')
    
    path_list = []
    dir_list = os.listdir(path)
    for dirname in dir_list:
        cur_path = os.path.join(path, dirname)
        
        if not os.path.isdir(cur_path):
            continue
        
        files = os.listdir(cur_path)

        if len(files) < num_files:
            path_list += [os.path.join(cur_path, filename) for filename in files]
    
    return path_list


# setting options
parser = OptionParser()
parser.add_option("-p", "--path", type="string",
                help="Root path having some directories to augmentate")
parser.add_option("-n", "--numfiles", type="int",
                help="select directories the number of each less than the value")

crop_option = OptionGroup(parser, "Random Crop Options",
                          "use these option at random crop function")
crop_option.add_option("-c", "--crop", default=True, action="store_false",
                help="replicates the random crop process")
crop_option.add_option("-r", "--crop_rate", default=0.7, 
                help="crop size: input image width * 0.7, height * 0.7")
parser.add_option_group(crop_option)

noise_option = OptionGroup(parser, "Noise Options",
                           "use these option at noise function")
noise_option.add_option("-g", "--noise", default=True, action="store_false",
                    help="replicates the image noising process")
noise_option.add_option("-i", "--noise_intensity", default=0.2,
                    help="use setting noise threshold: 1 - intensity")
parser.add_option_group(noise_option)


if __name__ == "__main__":
    # random_crop(['/home/mooooongni/사진/증명사진.jpg'])
    # noise_image(['/home/mooooongni/사진/증명사진.jpg'])
    (options, args) = parser.parse_args()

    path_list = select_dir(options.path, options.numfiles)    
    
    if options.crop:
        random_crop(path_list, options.crop_rate)
    if options.noise:
        noise_image(path_list, options.noise_intensity)
