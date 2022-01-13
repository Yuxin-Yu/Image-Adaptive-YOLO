import tensorflow as tf
from tensorflow.python.client import session
from stf import transformer
import numpy as np
import matplotlib.pyplot as plt
import imageio

tf.compat.v1.disable_eager_execution()

im=imageio.imread('/home/yyx/DeepLearning/Image-Adaptive-YOLO/data/foggyimages/SJZ_Bing_492.png')
im=im/255.
h,w,_=im.shape
im=tf.reshape(im, [1,h,w,3])

#im=im.reshape(1,1200,1600,3)
im = tf.compat.v1.Session().run(im)
im=im.astype('float32')
print('img-over')
out_size=(600,800)
batch=np.append(im,im,axis=0)
batch=np.append(batch,im,axis=0)
num_batch=3
 
x=tf.compat.v1.placeholder(tf.float32,[None,h,w,3])
x=tf.cast(batch,'float32')
print('begin---')

with tf.compat.v1.variable_scope('spatial_transformer_0'):
    n_fc=6
    w_fc1=tf.Variable(tf.Variable(tf.zeros([h*w*3,n_fc]),name='W_fc1'))
    initial=np.array([[0.5,0,0],[0,0.5,0]])
    initial=initial.astype('float32')
    initial=initial.flatten()
    
    
    b_fc1=tf.Variable(initial_value=initial,name='b_fc1')
    
    
    h_fc1=tf.matmul(tf.zeros([num_batch,h*w*3]),w_fc1)+b_fc1
    
    print(x,h_fc1,out_size)
 
    h_trans=transformer(x,h_fc1,out_size)
 
    
#sess=tf.compat.v1.Session()
#sess.run(tf.compat.v1.global_variables_initializer())
#y=sess.run(h_trans,feed_dict={x:batch})
#plt.imshow(y[0])
#plt.show()