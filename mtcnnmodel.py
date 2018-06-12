"""
mtcnn模型
"""
from . import netlayer

layer_params = [ [  10, 3, 1, 'valid', 'conv1', 'relu'], # pool
                 [  16, 3, 1, 'same',  'conv2', 'relu'], 
                 [  32, 3, 1, 'same',  'conv3', 'relu'], 
                 [  2, 1, 1, 'same',  'conv4-1', 'softmax'], 
                 [  4, 1, 1, 'same',  'conv4-2', 'relu'],
                 [  10,1, 1, 'same',  'conv4-3', 'relu']
               ]


def mtcnnconv(input,params,training):
    output = conv_layer(input, params[0],params[1],params[2],params[3],params[4],params[5], training ) # 30,30
    return output


def mtcnn_pnet(inputs, widths, mode):
    """Build convolutional network layers attached to the given input tensor"""

    training = (mode == learn.ModeKeys.TRAIN)

    with tf.variable_scope("mtcnn_pnet"): # h,w        
        conv1 = mtcnnconv(inputs, layer_params[0], training ) 
        pool1 = maxpool_layer(conv1,[2,2],2,'same','pool1')
        conv2 = mtcnnconv( pool1, layer_params[1], training ) 
        conv3 = conv_layer( conv2, layer_params[2], training ) 
        class_pret = conv_layer( conv3, layer_params[3], training ) 
        
        conv4_2 = conv_layer( pool4, layer_params[4], training ) # 7,14
        conv4_3 = conv_layer( conv5, layer_params[5], training ) # 7,14
        pool6 = pool_layer( conv6, 1, 'valid', 'pool6')        # 3,13
        conv7 = conv_layer( pool6, layer_params[6], training ) # 3,13
        conv8 = conv_layer( conv7, layer_params[7], training ) # 3,13
        pool8 = tf.layers.max_pooling2d( conv8, [3,1], [3,1], 
                                   padding='valid', name='pool8') # 1,13
        features = tf.squeeze(pool8, axis=1, name='features') # squeeze row dim

        kernel_sizes = [ params[1] for params in layer_params]

        # Calculate resulting sequence length from original image widths
        conv1_trim = tf.constant( 2 * (kernel_sizes[0] // 2),
                                  dtype=tf.int32,
                                  name='conv1_trim')
        one = tf.constant(1, dtype=tf.int32, name='one')
        two = tf.constant(2, dtype=tf.int32, name='two')
        after_conv1 = tf.subtract( widths, conv1_trim)
        after_pool2 = tf.floor_div( after_conv1, two )
        after_pool4 = tf.subtract(after_pool2, one)
        after_pool6 = tf.subtract(after_pool4, one) 
        after_pool8 = after_pool6

        sequence_length = tf.reshape(after_pool8,[-1], name='seq_len') # Vectorize

        return features,sequence_length