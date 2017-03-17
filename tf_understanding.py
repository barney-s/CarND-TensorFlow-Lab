import hashlib
import os
import pickle
import numpy as np
from urllib.request import urlretrieve
from PIL import Image

from sklearn.model_selection import train_test_split as sk_train_test_split
from sklearn.preprocessing import LabelBinarizer as sk_LabelBinarizer
from sklearn.utils import resample as sk_resample
from tqdm import tqdm
from zipfile import ZipFile
import math
import tensorflow as tf
import matplotlib.pyplot as plt


def download(url, file):
    """
    Download file from <url>
    :param url: URL to file
    :param file: Local file path
    """
    if not os.path.isfile(file):
        print('Downloading ' + file + '...')
        urlretrieve(url, file)
        print('Download Finished')
    else:
        print('Already exists .. skipping download... ' + file)


def uncompress_features_labels(file):
    """
    Uncompress features and labels from a zip file
    :param file: The zip file to extract the data from
    """
    features = []
    labels = []

    with ZipFile(file) as zipf:
        # Progress Bar
        filenames_pbar = tqdm(zipf.namelist(), unit='files')
        
        # Get features and labels from all files
        for filename in filenames_pbar:
            # Check if the file is a directory
            if not filename.endswith('/'):
                with zipf.open(filename) as image_file:
                    image = Image.open(image_file)
                    image.load()
                    # Load image data as 1 dimensional array
                    # We're using float32 to save on memory space
                    feature = np.array(image, dtype=np.float32).flatten()

                # Get the the letter from the filename.  This is the letter of the image.
                label = os.path.split(filename)[1][0]

                features.append(feature)
                labels.append(label)
    return np.array(features), np.array(labels)


def normalize_grayscale(image_data, a=0.1, b=0.9):
    """
    Normalize the image data with Min-Max scaling to a range of [0.1, 0.9]
    :param image_data: The image data to be normalized
    :return: Normalized image data
    """
    grayscale_min = 0
    grayscale_max = 255
    return a + ( ( (image_data - grayscale_min)*(b - a) )/( grayscale_max - grayscale_min ) )

def _normalize(feature, label):
    return normalize_grayscale(feature), label

def _encode(trnl, tstl):
    # Turn labels into numbers and apply One-Hot Encoding
    encoder = sk_LabelBinarizer()
    encoder.fit(trnl)
    return encoder.transform(trnl).astype(np.float32), encoder.transform(tstl).astype(np.float32)

def split_train_test(features, labels):
    #return:
    # train_features, valid_features, train_labels, valid_labels 
    tf, vf, tl, vl =  sk_train_test_split(features, labels, test_size=0.05, 
                            random_state=832289)
    return {
         'train_dataset': tf,
         'train_labels': tl,
         'valid_dataset': vf,
         'valid_labels': vf
     }

def prepare_data():
    sample_size = 150000
    # load features,labels  and Normalize features
    f, l = _normalize(*sk_resample(*uncompress_features_labels('notMNIST_train.zip'), n_samples=sample_size))
    tstf, tstl = _normalize(*uncompress_features_labels('notMNIST_test.zip'))
    # encode training and test labels
    l, tstl = _encode(l, tstl)
    # split training features into training and validation set
    sd = split_train_test(f, l)
    sd.update({'test_dataset': tstf, 'test_labels': tstl})
    return sd

def load_data():
    pickle_file = 'notMNIST.pickle'
    if not os.path.isfile(pickle_file):
        print('Saving data to pickle file...')
        with open('notMNIST.pickle', 'wb') as pfile:
            pickle.dump(prepare_data(), pfile, pickle.HIGHEST_PROTOCOL)

    # load cached data
    with open(pickle_file, 'rb') as pf:
        pickle_data = pickle.load(pf)
        trnf = pickle_data['train_dataset']
        trnl = pickle_data['train_labels']
        vf = pickle_data['valid_dataset']
        vl = pickle_data['valid_labels']
        tstf = pickle_data['test_dataset']
        tstl = pickle_data['test_labels']
        del pickle_data  # Free up memory
    return trnf, trnl, vf, vl, tstf, tstl


# - tf start --------------------------

def tf_pipeline():
    features_count = 784
    labels_count = 10
    features = tf.placeholder(tf.float32)
    labels = tf.placeholder(tf.float32)
    weights = tf.Variable(tf.truncated_normal((features_count, labels_count)))
    biases = tf.Variable(tf.zeros(labels_count))
    # Linear Function WX + b
    logits = tf.matmul(features, weights) + biases
    prediction = tf.nn.softmax(logits)
    # Cross entropy
    cross_entropy = -tf.reduce_sum(labels * tf.log(prediction), reduction_indices=1)
    # Training loss
    loss = tf.reduce_mean(cross_entropy)
    # Determine if the predictions are correct
    is_correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
    # Calculate the accuracy of the predictions
    accuracy = tf.reduce_mean(tf.cast(is_correct_prediction, tf.float32))
    # Create an operation that initializes all variables
    init = tf.global_variables_initializer()
    return init, loss, accuracy, features, labels


def iteration(name, epochs, batch_size, learning_rate, trnf, trnl, tstf, tstl):
    global init, loss, accuracy, features, labels
    train_feed_dict = {features: trnf, labels: trnl}
    test_feed_dict = {features: tstf, labels: tstl}
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    # The accuracy measured against the validation set
    # Measurements use for graphing loss and accuracy
    log_batch_step = 50
    batches = []
    loss_batch = []
    train_acc_batch = []
    valid_acc_batch = []
    print('---- type: {} epochs: {}, batch size: {}, learning rate: {} ----'.format(
                name, epochs, batch_size, learning_rate))
    with tf.Session() as session:
        session.run(init)
        batch_count = int(math.ceil(len(trnf)/batch_size))
        for epoch_i in range(epochs):
            # Progress bar
            batches_pbar = tqdm(range(batch_count), desc='Epoch {:>2}/{}'.format(epoch_i+1, epochs), unit='batches')
            # The training cycle
            for batch_i in batches_pbar:
                # Get a batch of training features and labels
                batch_start = batch_i*batch_size
                batch_features = trnf[batch_start:batch_start + batch_size]
                batch_labels = trnl[batch_start:batch_start + batch_size]
                # Run optimizer and get loss
                _, l = session.run([optimizer, loss], feed_dict={features: batch_features, labels: batch_labels})
                # Log every 50 batches
                if not batch_i % log_batch_step:
                    # Calculate Training and Validation accuracy
                    training_accuracy = session.run(accuracy, feed_dict=train_feed_dict)
                    _accuracy = session.run(accuracy, feed_dict=test_feed_dict)
                    # Log batches
                    previous_batch = batches[-1] if batches else 0
                    batches.append(log_batch_step + previous_batch)
                    loss_batch.append(l)
                    train_acc_batch.append(training_accuracy)
                    valid_acc_batch.append(_accuracy)
            # Check accuracy against Validation data
            _accuracy = session.run(accuracy, feed_dict=test_feed_dict)

    #loss_plot = plt.subplot(211)
    #loss_plot.set_title('Loss')
    #loss_plot.plot(batches, loss_batch, 'g')
    #loss_plot.set_xlim([batches[0], batches[-1]])
    #acc_plot = plt.subplot(212)
    #acc_plot.set_title('Accuracy')
    #acc_plot.plot(batches, train_acc_batch, 'r', label='Training Accuracy')
    #acc_plot.plot(batches, valid_acc_batch, 'x', label='Validation Accuracy')
    #acc_plot.set_ylim([0, 1.0])
    #acc_plot.set_xlim([batches[0], batches[-1]])
    #acc_plot.legend(loc=4)
    #plt.tight_layout()
    #plt.show()

    print('{} accuracy at {}'.format(name, _accuracy))
    

# -- main ------------------------------------------------------------

train_configs = [
    (1, 50, 0.1),
    (1, 300, 0.1),
    (1, 500, 0.1),
    (1, 1000, 0.1),
    (1, 2000, 0.1),
    (1, 100, 0.8),
    (1, 100, 0.5),
    (1, 100, 0.1),
    (1, 100, 0.05),
    (1, 100, 0.01),
    (1, 100, 0.2),
    (2, 100, 0.2),
    (3, 100, 0.2),
    (4, 100, 0.2),
    (5, 100, 0.2)
]

init, loss, accuracy, features, labels = tf_pipeline()
trnf, trnl, vf, vl, tstf, tstl = load_data()
for e, bs, lr in train_configs:
    iteration("Validation", e, bs, lr, trnf, trnl, vf, vl)
iteration("Test", 1, 50, 0.1, trnf, trnl, tstf, tstl)
