import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
from tqdm import tqdm
import time



# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    return image_input, keep_prob, layer3_out, layer4_out, layer7_out
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    # 1x1 convolution with vgg layer7 out
    # (-1, 5, 18, 4096)
    output = tf.layers.conv2d(vgg_layer7_out, 4096, 1, strides=(1,1))
        
    #upsampling to vgg layer 4
    # (-1, 10, 36, 512)
    output2 = tf.add(tf.layers.conv2d_transpose(output, 512, (2,2), (2,2)), 
                    vgg_layer4_out)
      
    #upsampling to vgg layer 3
    # (-1, 20, 72, 256)
    output3 = tf.add(tf.layers.conv2d_transpose(output2, 256, (2,2), (2,2), padding='VALID'),
                    vgg_layer3_out)
      
    #upsampling to (-1, 40, 144, 128)
    output4 = tf.layers.conv2d_transpose(output3, 128, (2,2), (2,2), padding='VALID')


    #FCL1_drop = tf.nn.dropout(FCL1, self.keep_prob)

    #upsampling to (-1, 80, 288, 32)
    output5 = tf.layers.conv2d_transpose(output4, 32, (2,2), (2,2), padding='VALID')
        
    #upsampling to (-1, 160, 576, 2)
    output6 = tf.layers.conv2d_transpose(output5, num_classes, (2,2), (2,2), padding='VALID')
    return output6
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function

    #flat = tf.reshape(output6, [-1, 2, 160*576])
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    
    #logits = tf.nn.softmax(flat)
    cross_entropy_lost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                              labels=correct_label))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_lost)
    return logits, train_op, cross_entropy_lost
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, image_input,
             correct_label, keep_prob, learning_rate, saver=None):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param image_input: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    start_t = time.time()
    for i in range(1,epochs+1):
        print('Iteration: {:02d} '.format(i), end="")
        images = get_batches_fn(batch_size)
        im, gt = next(images)
        #run_metadata = tf.RunMetadata()
        #res = sess.run([train_op, merged], feed_dict={image_input:im, lbls: gt, vgg_keep_prob:0.5}, run_metadata=run_metadata)
        [trn, loss] = sess.run([train_op, cross_entropy_loss], feed_dict={image_input:im, correct_label: gt, keep_prob:0.5})
        #tf.summary.FileWriter('./save', sess.graph)
        print('loss: {:02.4f} '.format(loss), end="")
        if (i % 10) == 0:
            print('\tSaving progress.')
            #run_metadata = tf.RunMetadata()
            #res = sess.run([merged], run_metadata=run_metadata)
            #tf.summary.FileWriter('./save', sess.graph)
            if saver != None:
                saver.save(sess, './fcn_save/save_net.ckpt')
        end_t = time.time()
        print('in {:02.4f} seconds.'.format(end_t - start_t))
        start_t = end_t
#tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # Build NN using load_vgg, layers, and optimize function
        [image_input, vgg_keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out] = load_vgg(sess, vgg_path)
        
        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)

        # Train NN using the train_nn function
        
        

        lbls = tf.placeholder(tf.float32, shape=[None, image_shape[0], image_shape[1], num_classes])
        correct_label = tf.reshape(lbls, (-1, num_classes))
        [logits, train_op, cross_entropy_loss] = optimize(nn_last_layer, correct_label, 1e-4, num_classes)



        saver = tf.train.Saver()

        # Check if network has previous parameters saved
        try:
            saver.restore(sess, './fcn_save/save_net.ckpt')
            print("Session checkpoint restored successfully!")
        except:
            print("No previous session checkpoint found.")
            init = tf.global_variables_initializer() 
            sess.run(init)
            pass

        
        train_nn(sess=sess, epochs=10, batch_size=4, get_batches_fn=get_batches_fn, train_op=train_op,
                    cross_entropy_loss=cross_entropy_loss, 
                    image_input=image_input,
                    correct_label=lbls,
                    keep_prob=vgg_keep_prob,
                    learning_rate=1e-4,
                    saver=saver)

        


        # TODO: Save inference data using helper.save_inference_samples
        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
        

        #helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, vgg_keep_prob, image_input)
        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
