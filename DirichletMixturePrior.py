import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers


class DirichletMixturePrior():
    def __init__(self, alpha, q):
        super(DirichletMixturePrior, self).__init__()
        # Dirichlet parameters > 0
        self.alpha = tf.constant(alpha)
        
        # mixture coefficients that sum to 1
        self.q = tf.constant(q)
        

    # in: n x alphabet_size count vectors 
    # out: n x k posterior probabilty distribution P(k | count)
    def posterior_component_prob(self, counts):
        
        #how many observed amino acids per column
        total_counts = tf.reduce_sum(counts, axis=-1, keepdims=True)
        
        #expand to allow broadcasting to shape n x k x alphabet_size
        alpha = tf.expand_dims(self.alpha,0)
        counts = tf.expand_dims(counts, 1)
        
        #obtain probabilities P(count | p_k) 
        dist = tfp.distributions.DirichletMultinomial(total_counts, alpha)
        probs = dist.prob(counts) #shape n x k
        
        #obtain P(component | count) via P(count | p_k) and the mix coefficients
        mix_probs = tf.expand_dims(self.q,0) * probs
        return mix_probs / tf.reduce_sum(mix_probs, axis=-1, keepdims=True)

    
    # in: n x alphabet_size count vectors 
    #     output of self.posterior_component_prob(count vectors)
    # out: n x alphabet_size posterior amino acid probabilty distributions
    def posterior_amino_acid_prob(self, counts, posterior_component_prob):
        
        counts = tf.expand_dims(counts, 1)
        alpha = tf.expand_dims(self.alpha, 0)
        total_counts = tf.reduce_sum(counts, axis=-1, keepdims=True)
        total_alpha = tf.reduce_sum(alpha, axis=-1, keepdims=True)
        pseudo_counts = (counts + alpha) / (total_counts + total_alpha)
        posterior_component_prob = tf.expand_dims(posterior_component_prob,1)
        #the (batch-)matmul is a marginalization of the component dimension k
        #shape posterior_component_prob : n x 1 x k
        #shape pseudo_counts            : n x k x alphabet_size
        probs_unnormalized = tf.matmul(posterior_component_prob, pseudo_counts) #shape n x 1 x alphabet_size
        Z = tf.reduce_sum(probs_unnormalized, axis=-1, keepdims=True)
        return tf.squeeze(probs_unnormalized / Z) #remove the 1-dimension
    
    
    # out: amino acid distribution of the j's mixture component
    def component_distributions(self):
        return self.alpha / tf.reduce_sum(self.alpha, axis=-1, keepdims=True)
    
    
def make_prior(alphabet):
    file_alphabet = "A C D E F G H I K L M N P Q R S T V W Y".split()

    q = []
    alpha = []

    k = 20

    with open("dirichlet_20comp.txt", "r") as file:
        content = file.readlines()
        for line in content:
            if line[:8] == "Mixture=":
                q.append(float(line[9:]))
            elif line[:6] == "Alpha=":
                alpha.append([float(a) for a in line[7:].split()][1:])

    q = np.array(q, dtype=np.float32) 
    alpha = np.array(alpha, dtype=np.float32) 


    #reorder alpha to make it consistent with the alphabet order used in other code
    perm = [file_alphabet.index(aa) for aa in alphabet[:20]] 
    alpha = alpha[:,perm]
    
    return DirichletMixturePrior(alpha, q)