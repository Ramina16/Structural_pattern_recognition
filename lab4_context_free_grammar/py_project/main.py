from PIL import Image

from functions import *

if __name__ == "__main__":

    image = Image.open('../images/example.png')
    zero = Image.open('../images/zero.png')
    one = Image.open('../images/one.png')

    img_tenzor = np.asarray(image)
    zero = np.asarray(zero)
    one = np.asarray(one)

    shape = img_tenzor.shape
    print(f' img shape {shape}')

    require_rest = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    leave_rest = np.array([0, 0, 1, 0, 1, 0, 1, 1])

    add_noise(img_tenzor, prob=0.3)
    labels, penalties = image_to_digits(img_tenzor, zero, one)
    penalties_matrix = find_all_penalties(penalties, imgs_x1=labels.shape[1],
                                          grammar_len=len(require_rest),
                                          require_rest=require_rest, leave_rest=leave_rest)
    markup = fing_markup(penalties_matrix, require_rest, leave_rest)
    final_img = digits_to_image(markup, zero, one)
    Image.fromarray(final_img).save("../images/markup.png")