import numpy as np
from math import sqrt
from typing import List

def printBoard(board):
    """
    Print current state of sudoku matrix
    :param board: n x n matrix with sudoku
    """
    print("    0 1 2     3 4 5     6 7 8")
    for i in range(len(board)):
        if i % 3 == 0:
            print("  - - - - - - - - - - - - - - ")
        for j in range(len(board[i])):

            label = ' ' if board[i][j] == 0 else board[i][j]
            if j % 3 == 0:
                print(" |  ", end="")
            if j == 8:
                print(label, " | ", i)
            else:
                print(f"{label} ", end="")
    print("  - - - - - - - - - - - - - - ")


def delete_unacceptable_marks(avaliability_matrix, i, j, mark):
    """
    Sets all the marks except chosen one to 0
    :param avaliability_matrix: n**2 x n**2 x n**2 matrix with available labels: every num in sudoku matrix
                                is represented as row 1 x n with 0 and 1, 0 corresponds to unavailable label,
                                1 - available ([0 0 0 0 0 1 0 0 1] = > available labels are [6, 9])
    :param i: idx of row
    :param j: idx of column
    :param mark: label
    """
    try:
        avaliability_matrix[i, j] = 0
        avaliability_matrix[i, j, mark - 1] = 1
    except IndexError as e:
        raise e


def find_neighbors(shape, i, j):
    """
    Finds neighbors of an object with coordinates (i,j)
    :param shape: shape of matrix in which neighbors will be found
    :param i: idx of row
    :param j: idx of column
    :return: list of all neighbors' coordinates
    """
    x_shape = shape[0]
    vertical_nb = [(k, j) for k in range(x_shape) if k != i]
    horisontal_nb = [(i, k) for k in range(x_shape) if k != j]

    sqrt_shape = int(sqrt(x_shape))
    x_cell_loc = [i // sqrt_shape * sqrt_shape + k for k in range(sqrt_shape)]
    y_cell_loc = [j // sqrt_shape * sqrt_shape + k for k in range(sqrt_shape)]
    cell_nb = [(n, m) for n in x_cell_loc for m in y_cell_loc if (n, m) != (i, j)]

    return vertical_nb + horisontal_nb + cell_nb, cell_nb


def delete_mark_in_neighbors(avaliability_matrix, i, j, mark):
    """
    If the mark is chosen in the object with coordinates (i,j)
    then this mark is removed from neighbors
    :param avaliability_matrix: n**2 x n**2 x n**2 matrix with available labels: every num in sudoku matrix
                                is represented as row 1 x n with 0 and 1, 0 corresponds to unavailable label,
                                1 - available  ([0 0 0 0 0 1 0 0 1] = > available labels are [6, 9])
    :param i: idx of row
    :param j: idx of column
    :param mark: label
    """
    neighbors, cell_nb = find_neighbors(avaliability_matrix.shape, i, j)
    for nb_x, nb_y in neighbors:
        avaliability_matrix[nb_x, nb_y, mark - 1] = 0


def remove_unacceptable_links(avaliability_matrix):
    """
    Delete labels from neighbors if on the position (i, j) is only one available label
    """
    for i in range(avaliability_matrix.shape[0]):
        for j in range(avaliability_matrix.shape[1]):
            if (len([k for k in avaliability_matrix[i, j] == 1 if k == True])) == 1:
                delete_mark_in_neighbors(avaliability_matrix, i, j, avaliability_matrix[i, j].argmax() + 1)


def set_mark(avaliability_matrix, i, j, mark):
    """
    Func set 'mark' to the object with coordinates (i,j)
    with all nessesary removement
    """
    if avaliability_matrix[i, j, mark - 1] != 0:
        delete_unacceptable_marks(avaliability_matrix, i, j, mark)
        delete_mark_in_neighbors(avaliability_matrix, i, j, mark)
        remove_unacceptable_links(avaliability_matrix)

        # print(avaliability_matrix)
    else:
        print(f'cannot set {mark} to matrix[{i},{j}]')


def initiate_structure(n: int, init_states: List = None):
    """
    Initiates sudoku matrix
    :param n: num of rows or cols in one square
    :param init_states: tuples (i, j, mark), optional
    :return: initial availability matrix
    """
    init_avaliability_matrix = np.ones(shape=(n ** 2, n ** 2, n ** 2), dtype=np.int32)

    if init_states:
        for state in init_states:
            init_matrix = set_mark(init_avaliability_matrix, *state)
    return init_avaliability_matrix


def print_martix(avaliability_matrix, verbose=True):
    """
    Transform avaliability_matrix to sudoku matrix
    :param avaliability_matrix: n**2 x n**2 x n**2 matrix with available labels: every num in sudoku matrix
                                is represented as row 1 x n with 0 and 1, 0 corresponds to unavailable label,
                                1 - available
    :param verbose: if True, print sudoku matrix
    :return: sudoku matrix
    """
    sudoku_matrix = np.zeros(shape=(avaliability_matrix.shape[:2]), dtype=np.int32)
    for i in range(avaliability_matrix.shape[0]):
        for j in range(avaliability_matrix.shape[1]):
            available_marks = len([k for k in avaliability_matrix[i, j] == 1 if k == True])
            if available_marks == 1:
                sudoku_matrix[i, j] = avaliability_matrix[i, j].argmax() + 1  # index+1 of 1, ex:(0,0,1,0)->3
                remove_unacceptable_links(avaliability_matrix)
            elif available_marks > 1:
                sudoku_matrix[i, j] = 0
            else:
                raise Exception(f"unavailable state ({i}, {j})")
    if verbose:
        printBoard(sudoku_matrix)

    return sudoku_matrix


def check_solved(matrix, n=9):
    """
    Check if sudoku already solved
    """
    return np.count_nonzero(matrix) == n**2


def sudoku_solver(init_sudoku, init_n=3, n=9):
    """
    Solve sudoku
    :param init_sudoku: n x n matrix with initial labels
    :paran init_n: num of rows or cols in one square
    :param n: num rows or num cols of sudoku matrix
    :return: matrix with solved sudoku
    """
    init_matrix = initiate_structure(init_n)
    for i in range(len(init_sudoku)):
        for j in range(len(init_sudoku[i])):
            if init_sudoku[i][j] == 0:
                continue
            set_mark(init_matrix, i, j, init_sudoku[i][j])
    res = print_martix(init_matrix)

    while not check_solved(res, n):
        for i in range(len(res)):
            for j in range(len(res[i])):
                if res[i, j] != 0:
                    continue
                labels = [k + 1 for k in range(len(init_matrix[i, j])) if init_matrix[i, j][k] == 1]
                set_mark(init_matrix, i, j, labels[0])
                res = print_martix(init_matrix)

    return res