from tensorflow.python.keras import *
import tensorflow as tf
import scipy.io as sio



Nt = 16  # the number of antennas
P = 1   # the normalized transmit power
K = 8   # number of users


def trans_Va(temp):
    v_real = tf.cos(temp)
    v_imag = tf.sin(temp)
    v_a = tf.cast(tf.complex(v_real, v_imag), tf.complex64)
    return v_a


def diff_func(temp):
    H , V_a = temp
    hva = backend.batch_dot(H, V_a)
    hva_herm = tf.transpose(tf.math.conj(hva),[0,2,1])
    T = tf.math.subtract(backend.batch_dot(hva_herm,hva),identity)
    diff = tf.norm(T,ord='euclidean')
    diff = tf.cast(diff , tf.float32)       
    return diff

