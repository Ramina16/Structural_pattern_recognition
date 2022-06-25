import numpy as np
import maxflow
import argparse
import os
import matplotlib.pyplot as plt

from itertools import product


def log_pdf(x, mu, cov):
    """
    :param x: numpy array of pixel RGB color layers numbers
    :param mu: distribution mean
    :param cov: distribution covariance matrix
    :return: log of probability density function for distribution with mu and cov
    """
    size = len(x)
    if size == len(mu) and (size, size) == cov.shape:
        det = np.linalg.det(cov)
        if det == 0:
            raise NameError("The covariance matrix can't be singular")
        norm_const = np.log(1.0 / ((2 * np.pi) ** (float(size) / 2) * det ** (1. / 2)))
        x_mu = x - mu
        inv = np.linalg.inv(cov)
        result = (- 0.5 * ((x_mu).dot(inv).dot(x_mu.T)))
        return norm_const + result
    else:
        raise NameError("The dimensions of the input don't match")


def log_likelyhood(pixel_color_list, segment_1_mean, segment_2_mean, segment_1_cov, segment_2_cov):
    """
    :param pixel_color_list: list of pixel RGB color layers
    :param segment_1_mean: distribution mean for segment 1
    :param segment_2_mean: distribution mean for segment 2
    :param segment_1_cov: distribution cov matrix for segment 1
    :param segment_2_cov: distribution cov matrix for segment 2
    :return: log probabilities of a pixel belonging to classes
    """
    pdf_1 = log_pdf(pixel_color_list, segment_1_mean, segment_1_cov)
    pdf_2 = log_pdf(pixel_color_list, segment_2_mean, segment_2_cov)

    return pdf_1, pdf_2


def find_neighbors_coord(i, j, dim1, dim2):
    """
    :param i: i position of a pixel (i,j)
    :param j: j position of a pixel (i,j)
    :param dim1: matrix height
    :param dim2: matrix width
    :return: list of tuples of neighbors coordinates for pixel (i,j)
    """
    neighbors = []
    for row_step in (-1, 0, 1):
        for col_step in (-1, 0, 1):
            if row_step * col_step == 0 and col_step != row_step:
                if i + row_step == -1 or j + col_step == -1:
                    continue
                if i + row_step >= dim1 or j + col_step >= dim2:
                    continue
                neighbors.append((i + row_step, j + col_step))

    return neighbors


def define_arcs_structure(matrix, eps, shape):
    """
    :param matrix: n x m x 2 matrix
    :param eps:  probability that the neighbor has different class
    :param shape: matrix's shape
    :return: n x m of neighbor structure for each pixel
    """
    arcs_structure = np.empty(shape=shape[:2], dtype=object)
    for i in range(shape[0]):
        for j in range(shape[1]):
            # for each pixel define empty structure
            arcs_structure[i, j] = {}
            neighbors = find_neighbors_coord(i, j, *shape[:2])
            for n_idx in neighbors:
                # on defined structure set neighbor's coord. as key and empty structure as value
                arcs_structure[i, j][n_idx] = {}
                for k_current, k_neigh in product((0, 1), repeat=2):
                    # update empty value. set a tuple of marks as a key and arc's weight as value
                    arcs_structure[i, j][n_idx][(k_current, k_neigh)] = (
                              (np.log(eps) if k_current != k_neigh else np.log(1 - eps))
                              + (matrix[i, j, k_current]) / (len(neighbors))
                              + (matrix[n_idx[0], n_idx[1]][k_neigh]) / (
                                  len(find_neighbors_coord(n_idx[0], n_idx[1], *shape[:2])))
                    )

    return arcs_structure


def add_graph_edges(max_flow_graph, nodes, arcs_structure, shape):
    """
    Fill maxflow.Graph with edges
    :param max_flow_graph: maxflow.GraphFloat() object
    :param nodes: n x m grid matrix from max_flow_graph where nodes[i][j] = num of el. in the matrix (1,..., n*m)
    :param arcs_structure: n x m structure where arcs_structure[n][m] define neighbors and edges with weights for pix [n][m];
                           edges are in the order (0,0), (0,1), (1,0), (1,1)
    :param shape: image shape (n x m)
    :return: max_flow_graph with updated edges
    """
    for i in range(shape[0]):
        for j in range(shape[1]):
            # neighborhood indices
            n_idx = list(arcs_structure[i][j].keys())
            neighbors_arcs = [dict(list(arcs_structure[i][j][t].items())) for t in arcs_structure[i][j]]
            # find list of lists where each el. in sublist is e.g. ((0,0), weight)
            values = [list(neighbors_arcs[l].items()) for l in range(len(neighbors_arcs))]
            # find edges values as (0,1) weight + (1,0) weight - (0,0) weight - (1,1) weight
            edges_values = [values[i][1][1] + values[i][2][1] - values[i][0][1] - values[i][3][1] for i in
                            range(len(values))]
            # add edge weight to corresponded position in max_flow_graph
            for n in range(len(n_idx)):
                max_flow_graph.add_edge(nodes[i, j], nodes[n_idx[n][0], n_idx[n][1]],
                                        edges_values[n], 0)
    return max_flow_graph


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_img', type=str)
    parser.add_argument('--class1_slice_idx', nargs='+', type=int,
                        help='4 idx corresponded to class1 segment: x_start, x_end, y_start, y_end')
    parser.add_argument('--class2_slice_idx', nargs='+', type=int,
                        help='4 idx corresponded to class2 segment: x_start, x_end, y_start, y_end')
    args = parser.parse_args()

    img_path = args.path_to_img

    class1_slice = args.class1_slice_idx
    class2_slice = args.class2_slice_idx

    img = plt.imread(img_path)
    img_shape = img.shape

    segment1 = img[class1_slice[2]:class1_slice[3], class1_slice[0]:class1_slice[1]]
    segment2 = img[class2_slice[2]:class2_slice[3], class2_slice[0]:class2_slice[1]]

    plt.figure()
    plt.imshow(segment1)
    plt.figure()
    plt.imshow(segment2)
    plt.show()

    segment1 = np.concatenate(segment1, axis=0)
    segment1_mean = segment1.mean(axis=0)
    segment1_cov = np.cov(segment1, rowvar=False)

    segment2 = np.concatenate(segment2, axis=0)
    segment2_mean = segment2.mean(axis=0)
    segment2_cov = np.cov(segment2, rowvar=False)

    image_data_probs = np.zeros(shape=(*img_shape[:2], 2), dtype=float)

    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            image_data_probs[i][j] = log_likelyhood(img[i][j], segment1_mean, segment2_mean, segment1_cov, segment2_cov)

    image_data_structure = define_arcs_structure(image_data_probs, 0.2, img_shape)

    mf_graph = maxflow.GraphFloat()
    nodes = mf_graph.add_grid_nodes(img_shape[:2])
    mf_graph.add_grid_tedges(nodes, image_data_probs[..., 0], image_data_probs[..., 1])
    max_flow_graph = add_graph_edges(mf_graph, nodes, image_data_structure, img_shape)
    flow = mf_graph.maxflow()
    mf_res = mf_graph.get_grid_segments(nodes)

    segmented_image = mf_res.astype('uint8') * 255

    path_folder = os.path.split(img_path)[0]
    name_result = os.path.basename(img_path).split('.')[0] + '_segmented.jpg'

    plt.imsave(os.path.join(path_folder, name_result), segmented_image, cmap='gray')
