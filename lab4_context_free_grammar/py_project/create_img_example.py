from PIL import Image
import numpy as np

from functions import  digits_to_image

one = Image.open('../images/one.png')
zero = Image.open('../images/zero.png')

one = np.asarray(one)
zero = np.asarray(zero)

example = np.array([[0,1,1,1,0,0],
                    [0,1,1,0,0,1],
                    [1,1,0,1,0,1]])

full_img = digits_to_image(example, zero, one)
Image.fromarray(full_img).save("../images/example.png")
