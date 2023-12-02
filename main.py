from PIL import Image

import numpy as np

if __name__ == '__main__':
    # load the image
    image = Image.open('test.png')

    # convert image to numpy array
    data = np.asarray(image)

    # remove alpha
    print(data[:, :, :-1].shape)
