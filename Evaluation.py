import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
import time
from sklearn.decomposition import PCA
from pylab import figure
from mpl_toolkits.mplot3d import Axes3D
import Config as config


##################################################################################################
##################################################################################################

def make_mask(y_true):
    mask = tf.math.equal(y_true, 0)
    mask = tf.math.reduce_all(mask, axis=-1)
    mask = tf.cast(tf.math.logical_not(mask), y_true.dtype)
    return mask

def make_sq(y, mask):
    y = tf.boolean_mask(y, mask)
    y_sq = tf.matmul(y, y, transpose_b=True)
    y_sq = tf.reshape(y_sq, (-1, 1))
    y_sq = tf.clip_by_value(y_sq, 0.0, 1.0)
    return y_sq

#precision and recall metrics for aligned aminoacid pairs

threshold = 0.5

def precision(y_true, y_pred):
    mask = make_mask(y_true)
    y_true_sq = make_sq(y_true, mask)
    y_pred_sq = make_sq(y_pred, mask)
    positives = tf.cast(y_pred_sq >= threshold, tf.float32) 
    true_positives = positives * y_true_sq
    precision = tf.reduce_sum(true_positives) / tf.math.maximum(tf.reduce_sum(positives), 1.0)
    return precision

def recall(y_true, y_pred):
    mask = make_mask(y_true)
    y_true_sq = make_sq(y_true, mask)
    y_pred_sq = make_sq(y_pred, mask)
    positives = tf.cast(y_pred_sq >= threshold, tf.float32)
    true_positives = positives * y_true_sq
    recall = tf.reduce_sum(true_positives) / tf.math.maximum(tf.reduce_sum(y_true_sq), 1.0)
    return recall

##################################################################################################
##################################################################################################

def plot_memberships(mem, outputs, titles, max_seq=20):
    
    num_seq = int(tf.shape(outputs[0])[0])
    num_outputs = len(outputs)
    
    plt.rcParams['figure.figsize'] = (16, 14)
    plt.subplots_adjust(hspace = .05, wspace=.05, left=0.05, right=0.95, top=0.98, bottom=0.02)
    fontsize = 10

    for i  in range(num_seq):
        if i == max_seq:
            break

        #plot the reference
        ax = plt.subplot(min(num_seq, max_seq), 1+num_outputs, (1+num_outputs)*i+1)
        if i == 0:
            ax.set_title("Reference", fontsize=fontsize)
        ax.get_xaxis().set_ticks([])
        ax.get_xaxis().set_ticklabels([])
        ax.get_yaxis().set_ticks([])
        ax.get_yaxis().set_ticklabels([])
        ax.set_ylabel("Sequence {}".format(i+1), rotation=0, fontsize=fontsize, labelpad=22+fontsize)
        plt.imshow(mem[i, :, :], cmap='hot', interpolation='nearest', aspect='auto') 

        #plot the prediction
        for j,(A,title) in enumerate(zip(outputs, titles)):
            ax = plt.subplot(min(num_seq, max_seq), 1+num_outputs, (1+num_outputs)*i + j + 2)
            if i == 0:
                ax.set_title(title, fontsize=fontsize)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.imshow(A[i, :, :], cmap='hot', interpolation='nearest', aspect='auto') 
        
    #plt.show()
    plt.savefig("./results/attention.pdf", bbox_inches='tight')