import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import time
from tqdm import tqdm 

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
    return None, None, None


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    pass


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        sess = tf.Session()
        [image_input, vgg_keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out] = load_vgg(sess, vgg_path)

        #operations = sess.graph.get_operations()
        #for op in operations:
        #    print(op.name)
       
        output6 = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)

      
        #flat = tf.reshape(output6, [-1, 2, 160*576])
        logits = tf.reshape(output6, (-1, num_classes))
        lbls = tf.placeholder(tf.float32, shape=[None, image_shape[0], image_shape[1], num_classes])
        labels = tf.reshape(lbls, (-1, num_classes))
        #logits = tf.nn.softmax(flat)

        cross_entropy_lost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                labels=labels))

        train_op = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_lost)

        #writer = tf.summary.FileWriter('./save', graph=tf.get_default_graph())
        #tensorboard --logdir=/logdir


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
        
#self.accuracy = tf.cast(self.mse, tf.float32)

#tf.scalar_summary('loss', self.mse)
# self.merged = tf.merge_all_summaries()#

#   self.train_writer = tf.train.SummaryWriter('./conv2/train',sess.graph)
#        self.test_writer = tf.train.SummaryWriter('./conv2/val')
#run_metadata = tf.RunMetadata()
#                summary, _, train_accuracy = sess.run([self.merged, self.train_step, self.accuracy], \
#                                                feed_dict={self.x: images, self.y_: angles, self.keep_prob: 0.5}, \
#                                                    run_metadata=run_metadata)


        #tf.summary.scalar('loss', cross_entropy_lost)
        #merged = tf.summary.merge_all()


        #start_t = time.time() 
        for _ in tqdm(range(100), desc='Iteration'):
            #print('Iteration: {:02d} '.format(_+1), end="")
            images = get_batches_fn(4)
            im, gt = next(images)
            #run_metadata = tf.RunMetadata()
            #res = sess.run([train_op, merged], feed_dict={image_input:im, lbls: gt, vgg_keep_prob:0.5}, run_metadata=run_metadata)
            res = sess.run([train_op], feed_dict={image_input:im, lbls: gt, vgg_keep_prob:0.5})
            #tf.summary.FileWriter('./save', sess.graph)
            if (_ % 10) == 0:
                #print('\tSaving progress.')
                #run_metadata = tf.RunMetadata()
                #res = sess.run([merged], run_metadata=run_metadata)
                #tf.summary.FileWriter('./save', sess.graph)
                saver.save(sess, './fcn_save/save_net.ckpt')
            #end_t = time.time()
            #print('in {:02.4f} seconds.'.format(end_t - start_t))
            #start_t = end_t
 

        # TODO: Train NN using the train_nn function

        # TODO: Save inference data using helper.save_inference_samples
        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
        

        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, vgg_keep_prob, image_input)
        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
