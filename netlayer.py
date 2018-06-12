
""" 
＃定义卷积神经网络单元
"""  
import tensorflow as tf
from tensorflow.contrib import learn


"""
＃卷积层代码
param input: 输入层数据
param filternum: 卷积核数量　
param kernelsize:卷积核大小
param strides:卷积步长
param pad: 张量填充方式
param scope:　层名
activation: 1.'bn' 使用batchnorm层　则后面默认接relu激活　２．'relu' 3. 'softmax'：　softmax
param training: 是否训练阶段

output:　卷积层输出
"""
def conv_layer(input,filternum,kernelsize,strides,pad,scope,activation,training):
    if activation == 'bn':
        activation_fn = None
    elif activation == 'relu':
        activation_fn = tf.nn.relu
    elif activation == 'softmax':
        activation_fn = tf.nn.softmax
    else:
        activation_fn = tf.nn.relu

    #参数的初始化方式
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    bias_initializer = tf.constant_initializer(value=0.0)

    #tf卷积层
    output = tf.layers.conv2d(input, 
                        filters=filternum,
                        kernel_size=kernelsize,
                        strides = strides,
                        padding=pad,
                        activation=activation,
                        kernel_initializer=kernel_initializer,
                        bias_initializer=bias_initializer,
                        name=scope,
                        trainable=training )
    
    if activation == 'bn':
        norm = batchnorm_layer(output,scope + '/batchnorm',training)
        output = tf.nn.relu(norm,name = scope + '/relu')

    return output



"""
＃batchnorm层代码
param input: 输入层数据
param scope:　层名
param training: 是否训练阶段

output:　批归一化输出
"""
def batchnorm_layer(input, scope,training,):
    norm = tf.layers.batch_normalization(input,name = scope,  training=training)
    return norm



"""
＃最大池化层
param input: 输入层数据
param poolsize: 池化核数量　
param strides:池化步长
param pad: 张量填充方式
param scope:　层名

output:　最大池化层输出
"""
def maxpool_layer( input, poolsize,strides, pad, scope ): 

    output = tf.layers.max_pooling2d( input, 
                                   pool_size=poolsize,
                                   strides=strides, 
                                   padding=pad, 
                                   name=scope)
    return output