import tensorflow as tf
import tf_slim

def ASPP(inputs, dilated_series, output_depth):
        """
        Implementation of the Atrous Spatial Pyramid Pooling described of DeepLabv3.
        :param inputs: A Tensor of size [batch, height_in, width_in, channels].
        :param dilated_series: A tuple of the atrous rate.
        :param output_depth: The output depth of the layer.
        :return:
            aspp_list: A list contain the feature map Tensor after aspp.
        """
        with tf.compat.v1.variable_scope("aspp"):
            aspp_list = []
            branch_1 = tf_slim.conv2d(inputs, num_outputs=output_depth, kernel_size=1, stride=1, scope="conv_1x1")
            aspp_list.append(branch_1)

            for i in range(3):
                branch_2 = tf_slim.conv2d(inputs, num_outputs=output_depth, kernel_size=3, stride=1, rate=dilated_series[i],
                                       scope="rate{}".format(dilated_series[i]))
                aspp_list.append(branch_2)

            return aspp_list