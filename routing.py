import tensorflow as tf
import numpy as np


def squash(tensor):
    '''Squashing function.
    Args:
        tensor: A 5-D tensor with shape [batch_size, h, w, caps_dim, num_node],
    Returns:
        A 5-D tensor with the same shape as vector but
        squashed in 3rd and 4th dimensions.
    '''
    tensor_norm = tf.sqrt(tf.reduce_sum(tf.square(tensor))) # length(scalar)
    scale_factor = tf.square(tensor_norm) / (1 + tf.square(tensor_norm))
    tensor_squashed = tf.multiply(scale_factor, tf.divide(tensor, tensor_norm))
    return tensor_squashed

# version1, we use every "grid"(node) in the lower layer to predict every grid(node) in the upper layer
# (grid-wise, not capsule-wise)
# based on the prediction, we update the pair-coefficients, then we get our beta_capsule_grid

# we need a new routing function that not only helps us update the pair-coefficients,
# but also give us the beta_prediction (which stack the two nodes' prediction together)


def alpha_infer(input, num_outputs, cap_dim, kernel_size=None, stride=None):
    capsules = tf.contrib.layers.conv2d(input, num_outputs * cap_dim,
                                            kernel_size, stride, padding="VALID",
                                            activation_fn=tf.nn.relu)
    capsules = squash(capsules)
    return capsules


def beta_routing(lower_cap, upper_cap, lower_num, upper_num, b_ij, num_routing=3):
    """
    This implements the "inference"-determined data routing algorithm
    :param lower_cap: tensor
    :param upper_cap: tensor
    :param lower_num: number of "grids" of capsules in the lower layer
    :param upper_num: number of "grids" of capsules in the lower layer
    :param b_ij: the logits of pair coefficients c_ij
    :param num_routing: the routing times to update b_ij
    :return: our beta prediction for the upper_cap
    """
    b_l, h_l, w_l, d_l = lower_cap.shape
    b_u, h_u, w_u, d_u = upper_cap.shape
    assert b_l == b_u  # batchSize should be the same
    assert h_l == h_u
    assert w_l == w_u
    batchS = b_l
    h = h_l
    w = w_l
    lower_cap_dim = d_l / lower_num
    upper_cap_dim = d_u / upper_num
    # Every lower cap's node(gird)'s shape is (h_l, w_l, lower_cap_dim), and there are 8(2) of them
    # Every upper cap's node(grid)'s shape is (h_u, w_u, upper_cap_dim), and there are 2(1) of them
    # We need a 3D matrix(with batchSize times, so a 4D tensor) to do the "inference"
    lower_cap_nodes = []
    upper_cap_nodes = []
    for i in range(lower_num):
        lower_cap_nodes.append(lower_cap[:, :, :, lower_cap_dim * i:lower_cap_dim * (i + 1) - 1])
        # every node's shape is (b_l, h_l, w_l, lower_cap_dim)
    for i in range(upper_num):
        upper_cap_nodes.append(upper_cap[:, :, :, upper_cap_dim * i:upper_cap_dim * (i + 1) - 1])
        # every node's shape is (b_u, h_u, w_u, upper_cap_dim)

    w_initializer = np.random.normal(size=[1, lower_num, upper_num, 28, 28, lower_cap_dim, upper_cap_dim], scale=0.01)
    W_Ij = tf.Variable(w_initializer, dtype=tf.float32)
    W_Ij = tf.tile(W_Ij, [batchS, 1, 1, 1, 1, 1, 1])
    # predictions are made by tf.matmul
    predictions = []
    for i in range(lower_num):
        for j in range(upper_num):
            l_node = lower_cap_nodes[i]
            l_node = tf.reshape(l_node, (batchS, h, w, lower_cap_dim, 1))
            weight = W_Ij[:, i, j, :, :, :]
            weight = tf.reshape(weight, (batchS, h, w, lower_cap_dim, upper_cap_dim))
            pred_node = tf.matmul(weight, l_node, transpose_a=True)
            pred_node = tf.reshape(pred_node, (batchS, h, w, upper_cap_dim))
            pred_node = squash(pred_node)
            predictions.append(pred_node)

    # Using the predictions to determine c_ij
    b_ij = tf.tile(b_ij, [batchS, 1, 1])
    for i in range(num_routing):
        for k in range(lower_num):
            if upper_num == 2:
                pred1 = predictions[2 * k]
                pred2 = predictions[2 * k + 1]
                up_node1 = upper_cap_nodes[2 * k]
                up_node2 = upper_cap_nodes[2 * k + 1]
                assert pred1.shape == up_node1.shape
                assert pred2.shape == up_node2.shape
                inference1 = tf.reduce_sum(tf.multiply(pred1, up_node1), axis=[1, 2, 3])
                inference2 = tf.reduce_sum(tf.multiply(pred2, up_node2), axis=[1, 2, 3])
                # now inference1 and inference2 should both have the shape (batchS,)
                # i.e. for every image, we calculate an inference score for nodei to upper node1 and upper node2
                b_ij[batchS, k, 0] += inference1
                b_ij[batchS, k, 1] += inference2
                # when upper_num == 1, no need to update b_ij for beta, it's always 0

    # create pair_coefficient c_ij
    c_ij = tf.nn.softmax(b_ij)
    # now c_i1 + c_i2 = 1
    beta_upper_nodes = []
    for j in range(upper_num):
        pred_node = tf.zeros((batchS, h_l, w_l, upper_cap_dim))
        for i in range(lower_num):
            pred_node += tf.multiply(c_ij[batchS, i, j], predictions[i * upper_num + j])
        pred_node = squash(pred_node)
        beta_upper_nodes.append(pred_node)
    # finally we use the pair coefficients to give pred_upper_cap using W and lower_cap
    beta_upper_cap = tf.concat(beta_upper_nodes, 3)
    assert beta_upper_cap.shape == part_body_land_outs.shape
    return beta_upper_cap


def gamma_routing(upper_cap, lower_cap, upper_num, lower_num, b_ij, num_routing=3):
        """
        This implements the "inference"-determined data routing algorithm
        :param lower_cap: tensor
        :param upper_cap: tensor
        :param lower_num: number of "grids" of capsules in the lower layer
        :param upper_num: number of "grids" of capsules in the lower layer
        :param b_ij: the logits of pair coefficients c_ij
        :param num_routing: the routing times to update b_ij
        :return: our beta prediction for the lower_cap
        """
        b_l, h_l, w_l, d_l = lower_cap.shape
        b_u, h_u, w_u, d_u = upper_cap.shape
        assert b_l == b_u  # batchSize should be the same
        assert h_l == h_u
        assert w_l == w_u
        batchS = b_l
        h = h_l
        w = w_l
        lower_cap_dim = d_l / lower_num
        upper_cap_dim = d_u / upper_num
        # Every lower cap's node(gird)'s shape is (h_l, w_l, lower_cap_dim), and there are 8(2) of them
        # Every upper cap's node(grid)'s shape is (h_u, w_u, upper_cap_dim), and there are 2(1) of them
        # We need a 3D matrix(with batchSize times, so a 4D tensor) to do the "inference"
        lower_cap_nodes = []
        upper_cap_nodes = []
        for i in range(lower_num):
            lower_cap_nodes.append(lower_cap[:, :, :, lower_cap_dim * i:lower_cap_dim * (i + 1) - 1])
            # every node's shape is (batchS, h_l, w_l, lower_cap_dim)
        for i in range(upper_num):
            upper_cap_nodes.append(upper_cap[:, :, :, upper_cap_dim * i:upper_cap_dim * (i + 1) - 1])
            # every node's shape is (b_u, h_u, w_u, upper_cap_dim)

        w_initializer = np.random.normal(size=[1, upper_num, lower_num, 28, 28, upper_cap_dim, lower_cap_dim],
                                         scale=0.01)
        W_Ij = tf.Variable(w_initializer, dtype=tf.float32)
        W_Ij = tf.tile(W_Ij, [batchS, 1, 1, 1, 1, 1, 1])
        # predictions are made by tf.matmul
        predictions = []
        for i in range(upper_num):
            for j in range(lower_num):
                l_node = upper_cap_nodes[i]
                l_node = tf.reshape(l_node, (batchS, h, w, upper_cap_dim, 1))
                weight = W_Ij[:, i, j, :, :, :]
                weight = tf.reshape(weight, (batchS, h, w, upper_cap_dim, lower_cap_dim))
                pred_node = tf.matmul(weight, l_node, transpose_a=True)
                pred_node = tf.reshape(pred_node, (batchS, h, w, lower_cap_dim))
                pred_node = squash(pred_node)
                predictions.append(pred_node)

        # Using the predictions to determine c_ij
        b_ij = tf.tile(b_ij, [batchS, 1, 1])
        for i in range(num_routing):
            for k in range(upper_num):
                for j in range(lower_num):
                    pred = predictions[j + k * lower_num]
                    low_node = lower_cap_nodes[j + k * lower_num]
                    assert pred.shape == low_node.shape
                    inference = tf.reduce_sum(tf.multiply(pred, low_node), axis=[1, 2, 3])
                    b_ij[batchS, k, j] += inference

        # create pair_coefficient c_ij
        c_ij = tf.nn.softmax(b_ij)
        # now c_i1 + c_i2 = 1
        gamma_lower_nodes = []
        for j in range(lower_num):
            pred_node = tf.zeros((batchS, h, w, lower_cap_dim))
            for i in range(upper_num):
                pred_node += tf.multiply(c_ij[batchS, i, j], predictions[i * lower_num + j])
            pred_node = squash(pred_node)
            gamma_lower_nodes.append(pred_node)

        gamma_lower_cap = tf.concat(gamma_lower_nodes, 3)
        assert gamma_lower_cap.shape == lower_cap.shape
        return gamma_lower_cap






# class CapsConv(object):
#     """
#     Capsule layer. Used in alpha-beta-gamma only to transfer conv_5_out to a alpha
#     Args:
#         input: A 4-D tensor.
#         cap_dim: integer, capsule_dim
#         num_outputs: number of "grids" of capsules
#     Returns:
#         A 4-D tensor.
#     """
#
#     def __init__(self, num_outputs, cap_dim):
#         self.cap_dim = cap_dim
#         self.num_outputs = num_outputs
#
#     def __call__(self, input, kernel_size=None, stride=None):
#         """
#         Change the input to a capsule layer
#         :param input: conv layer input
#         :param kernel_size: kernel size
#         :param stride: stride
#         :return: a capsule layer, the alpha inference result
#         """
#         self.kernel_size = kernel_size
#         self.stride = stride
#
#         # batchS = input.get_shape().as_list()[0]
#         capsules = tf.contrib.layers.conv2d(input, self.num_outputs * self.cap_dim,
#                                             self.kernel_size, self.stride, padding="VALID",
#                                             activation_fn=tf.nn.relu)
#         capsules = squash(capsules)
#         return capsules



