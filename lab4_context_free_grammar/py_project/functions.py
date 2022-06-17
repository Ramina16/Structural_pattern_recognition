import numpy as np


def to_binary(img):
    binary = np.zeros(img.shape[:2], dtype=np.int8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j].all() > 0:
                binary[i, j] = 0
            else:
                binary[i, j] = 255


def to_bw(img):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j].all() > 0:
                img[i, j] = [0, 0, 0]
            else:
                img[i, j] = [255, 255, 255]


def digits_to_image(matrix, zero, one):
    lines = []
    for i in range(matrix.shape[0]):
        items = []
        for j in range(matrix.shape[1]):
            if matrix[i, j] == 0:
                items.append(zero)
            elif matrix[i, j] == 1:
                items.append(one)
            else:
                raise Exception(f"undefined digit on ({i},{j})")
        lines.append(np.concatenate(items, axis=1))
    full_img = np.concatenate(lines, axis=0)
    return full_img


def image_to_digits(image, zero, one):
    x_step = zero.shape[0]
    y_step = zero.shape[1]
    x_len = image.shape[0] // zero.shape[0]
    y_len = image.shape[1] // zero.shape[1]
    print(f"vertical steps: {x_len}, horisontal steps {y_len}")
    image_names = np.zeros((x_len, y_len), dtype=np.int8)
    penalties = np.zeros((x_len, y_len, 2), dtype=np.int16)
    for x in range(x_len):
        for y in range(y_len):
            penalty0 = 0
            penalty1 = 0
            for i in range(zero.shape[0]):
                for j in range(zero.shape[1]):
                    if image[x * x_step + i, y * y_step + j].any() != zero[i, j].any():
                        penalty0 += 1
                    if image[x * x_step + i, y * y_step + j].any() != one[i, j].any():
                        penalty1 += 1
            image_names[x, y] = (1 if penalty1 < penalty0 else 0)
            penalties[x, y] = [penalty0, penalty1]
    return image_names, penalties


def add_noise(array, prob=0.2):
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            array[i, j] = (array[i, j] + np.random.choice([255, 0], size=1, p=[prob, 1 - prob])) % 510


def find_all_penalties(penalties, imgs_x1, grammar_len, require_rest, leave_rest):
    # 1 find penalties for each column case
    full_img_penalties = np.zeros((imgs_x1, grammar_len), dtype=float)
    for i in range(full_img_penalties.shape[0]):
        for j in range(full_img_penalties.shape[1]):
            lower_k = j % 2
            middle_k = (j // 2) % 2
            higher_k = (j // 4) % 2
            full_img_penalties[i, j] = (
                    penalties[2, i, lower_k] +
                    penalties[1, i, middle_k] +
                    penalties[0, i, higher_k]
            )
    # 2 find available for the last column
    for j, val in enumerate(require_rest):
        if val == 1:
            full_img_penalties[-1, j] = np.inf
    # 3 find available for the last column
    for j, val in enumerate(leave_rest):
        if val == 1:
            full_img_penalties[0, j] = np.inf

    return full_img_penalties


def fing_markup(penalties_matrix, require_rest, leave_rest, verbose=2):
    idx_leave_rest = np.where(leave_rest == 1)
    idx_not_leave_rest = np.where(leave_rest == 0)
    for i in range(penalties_matrix.shape[0] - 2, -1, -1):
        prev_penalty_leave_rest = penalties_matrix[i + 1, idx_leave_rest].min()
        prev_penalty_not_leave_rest = penalties_matrix[i + 1, idx_not_leave_rest].min()
        for this_col in range(penalties_matrix.shape[1]):
            if require_rest[this_col] == 1:
                penalties_matrix[i, this_col] = penalties_matrix[i, this_col] + prev_penalty_leave_rest
            if require_rest[this_col] == 0:
                penalties_matrix[i, this_col] = penalties_matrix[i, this_col] + prev_penalty_not_leave_rest
    if verbose == 2:
        print(f"\npenalties_matrix\n {penalties_matrix}")

    min_penalty_idx = []
    for i in range(penalties_matrix.shape[0]):
        if np.min(penalties_matrix[i, :]) != np.inf:
            min_penalty_idx.append(np.argmin(penalties_matrix[i, :]))
        else:
            raise Exception("incorrect input example!")
    if verbose >= 1:
        print(f"\nmin_penalty_idx\n {min_penalty_idx}")

    result_matrix = np.zeros((3, len(min_penalty_idx)))
    for i in range(result_matrix.shape[1]):
        result_matrix[2, i] = min_penalty_idx[i] % 2
        result_matrix[1, i] = (min_penalty_idx[i] // 2) % 2
        result_matrix[0, i] = (min_penalty_idx[i] // 4) % 2

    return result_matrix