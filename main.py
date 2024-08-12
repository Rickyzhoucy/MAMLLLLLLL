import numpy as np
import tensorflow as tf
import random
import pickle
import csv
from data_generator import DataGenerator
from maml import MAML

# Set up flags
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('datasource', 'sinusoid', 'Type of data: sinusoid, omniglot, or miniimagenet')
tf.app.flags.DEFINE_integer('num_classes', 5, 'Number of classes for classification (e.g., 5-way classification).')
tf.app.flags.DEFINE_integer('metatrain_iterations', 15000, 'Number of meta-training iterations.')
tf.app.flags.DEFINE_integer('meta_batch_size', 25, 'Number of tasks sampled per meta-update.')
tf.app.flags.DEFINE_float('meta_lr', 0.001, 'Meta learning rate.')
tf.app.flags.DEFINE_integer('update_batch_size', 5, 'Number of examples used for inner gradient updates.')
tf.app.flags.DEFINE_float('update_lr', 1e-3, 'Learning rate for inner updates.')
tf.app.flags.DEFINE_integer('num_updates', 1, 'Number of inner gradient updates during training.')
tf.app.flags.DEFINE_bool('train', True, 'True to train, false to test.')

def initialize_session():
    """Initialize session and model saver."""
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=10)
    return sess, saver

def training_phase(model, saver, sess, data_generator):
    """Training loop for the model."""
    for itr in range(FLAGS.metatrain_iterations):
        batch_x, batch_y, amp, phase = data_generator.generate()
        inputa = batch_x[:, :FLAGS.num_classes*FLAGS.update_batch_size, :]
        labela = batch_y[:, :FLAGS.num_classes*FLAGS.update_batch_size, :]
        inputb = batch_x[:, FLAGS.num_classes*FLAGS.update_batch_size:, :]
        labelb = batch_y[:, FLAGS.num_classes*FLAGS.update_batch_size:, :]

        feed_dict = {model.inputa: inputa, model.inputb: inputb, model.labela: labela, model.labelb: labelb}
        sess.run([model.metatrain_op], feed_dict)

        if itr != 0 and itr % 100 == 0:
            print(f'Iteration {itr}: Loss = {sess.run(model.total_losses2[-1], feed_dict)}')

        if itr != 0 and itr % 1000 == 0:
            saver.save(sess, FLAGS.logdir + '/model', global_step=itr)

def main():
    data_generator = DataGenerator(FLAGS.update_batch_size*2, FLAGS.meta_batch_size)
    model = MAML(dim_input=data_generator.dim_input, dim_output=data_generator.dim_output)
    model.construct_model()

    sess, saver = initialize_session()

    if FLAGS.train:
        training_phase(model, saver, sess, data_generator)
    else:
        testing_phase(model, sess, data_generator)  # Define this function if needed for testing

if __name__ == '__main__':
    main()
