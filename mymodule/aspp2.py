#################ASPP

import tensorflow as tf

tf.compat.v1.disable_eager_execution()

x = tf.compat.v1.placeholder(tf.float32,[None, 500, 500, 3])#输入图片大小

def dilated_conv_layer(x, shape, dilation,name):
    '''
    相当于CNN中的卷积核，要求是一个4维Tensor，具有[filter_height, filter_width, channels, out_channels]这样的shape，具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]，同理这里第三维channels，就是参数value的第四维
    '''
    # filter表示卷积核的构造
    filters = tf.compat.v1.get_variable(name,
                              shape=shape,
                              dtype=tf.float32,
                              initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                              trainable=True)
    # 进行空洞卷积，dilation表示卷积核补零的大小
    return tf.nn.atrous_conv2d(x, filters, dilation, padding='SAME')
def filter_tensor(shape,name):
    return tf.compat.v1.get_variable(name,  # filter表示卷积核的构造
                              shape=shape,
                              dtype=tf.float32,
                              initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                              trainable=True)

                              
def ASPP2(x, rate1, rate2, rate3, rate4, channel):
    ##第一层
    layer1_1=dilated_conv_layer(x,[500,500,256,channel],rate1,name='filter1')
    layer1_2=dilated_conv_layer(x,[500,500,256,channel],rate2,name='filter2')
    layer1_3=dilated_conv_layer(x,[500,500,256,channel],rate3,name='filter3')
    layer1_4=dilated_conv_layer(x,[500,500,256,channel],rate4,name='filter4')
    ##第二层
    layer1_1=tf.nn.conv2d(layer1_1,filter_tensor([500,500,channel,channel],name= 'filter_tensor1'),strides=1, padding='same')
    layer1_2=tf.nn.conv2d(layer1_2,filter_tensor([500,500,channel,channel],name= 'filter_tensor2'),strides=1, padding='same')
    layer1_3=tf.nn.conv2d(layer1_3,filter_tensor([500,500,channel,channel],name= 'filter_tensor3'),strides=1, padding='same')
    layer1_4=tf.nn.conv2d(layer1_4,filter_tensor([500,500,channel,channel],name= 'filter_tensor4'),strides=1, padding='same')
    #第三层
    layer1_1=tf.nn.conv2d(layer1_1,filter_tensor([500,500,channel,channel],name ='filter_tensor5'),strides=1, padding='same')
    layer1_2=tf.nn.conv2d(layer1_2,filter_tensor([500,500,channel,channel],name ='filter_tensor6'),strides=1, padding='same')
    layer1_3=tf.nn.conv2d(layer1_3,filter_tensor([500,500,channel,channel],name ='filter_tensor7'),strides=1, padding='same')
    layer1_4=tf.nn.conv2d(layer1_4,filter_tensor([500,500,channel,channel],name ='filter_tensor8'),strides=1, padding='same')     
    
    output=layer1_1+layer1_2+layer1_3+layer1_4
    return output
   
layer1=tf.nn.conv2d(x,filter_tensor([500,500,3,255],name ='filter_tensor9'),strides=1, padding='same')
layer1=tf.nn.relu(layer1)
layer2=ASPP2(layer1,2,4,6,8,256)
print(layer2)
