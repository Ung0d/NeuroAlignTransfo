import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import numpy as np
import Config as config
import Data as data



# Implementation of the NeuroAlign model
# Here we use a Transformer sequence to sequence model.
# The code is based on https://www.tensorflow.org/tutorials/text/transformer.
# The main idea is to transform each sequence individually to the
# sequence of columns and aggregate thereafter over the number of sequences. 
# This procedure is repeated over multiple iterations to allow each input sequence to
# reason about its own contribution to the consensus relative to all other sequences.
# Eventually, the model outputs aggregated columns and attention weights (i.e. a number between 
# 0 and 1 for each pair of residuum and column)


#a mask for a 3D tensor where the 2nd dimension is the spatial/temporal dimension
#mask is 1 if and only if a position is non padded (and thus can attend)
#a padded position is a sequence position where the last dimension is all zero
def make_padding_mask(sequences):
    mask = tf.math.equal(sequences, 0)
    mask = tf.math.reduce_all(mask, axis=-1)
    mask = tf.cast(tf.math.logical_not(mask), sequences.dtype)
    return tf.reshape(mask, (-1, 1, 1, tf.shape(sequences)[1]))

#a look ahead mask indicates for each element in a target sequence, which elements (here all previous) should attend
def make_look_ahead_mask(x):
    num = tf.shape(x)[-2]
    mask = tf.linalg.band_part(tf.ones((num, num)), -1, 0)
    return mask  # (num, num)

# experimental and currently unused
# computes a dynamic mask that depends on the attention matrix of the 
# sequences to the columns using cumulative probabilities
# if a column j-1 attends to a position i, then column j can only attend to
# all positions i+1 ... L (colinearity)
# input: (B, num_cols, len_seq) prob. dist over len_seq
def make_dynamic_colinear_mask(A, mask):
    col_mask = tf.math.cumsum(A, axis=-1, exclusive=True)
    
    #col_mask = tf.math.cumprod(col_mask, axis=-2, exclusive=True)
    
    col_mask = tf.roll(col_mask, shift=1, axis=-2)
    oh = tf.reshape(tf.one_hot([0], depth = tf.shape(col_mask)[-2]), (1, 1, -1, 1))
    col_mask = col_mask * (1 - oh) + oh #replaces col_mask[:,0,:] with ones
    
    return col_mask 

##################################################################################################
##################################################################################################

#implements multi-head attention
#we use scaled dot product attention for efficiency
class MultiHeadAttention(layers.Layer):
    
    def __init__(self, dim, heads):
        super(MultiHeadAttention, self).__init__()
        self.heads = heads
        self.dim = dim

        assert dim % heads == 0

        self.head_dim = dim // heads

        self.wq = layers.Dense(dim)
        self.wk = layers.Dense(dim)
        self.wv = layers.Dense(dim)
        
        self.dense = layers.Dense(dim)
        self.SCALE = 1e4
        
        
    def scaled_dot_product_attention(self,
                                     query,  #(..., Q, D1)
                                     keys,   #(..., KV, D1)
                                     values, #(..., KV, D2)
                                     mask = None):  #(..., Q, KV) (or broadcastable to that shape)

        QK = tf.matmul(query, keys, transpose_b=True) #(...,Q,KV)
        QK /= tf.math.sqrt(tf.cast(tf.shape(keys)[-1], keys.dtype))
        if mask != None:
            QK += ((1-mask) * -1e9)  #masked positions will be close to zero after softmaxing
        A = tf.nn.softmax(QK, axis=-1)  
        out = tf.matmul(A, values)  #(...,Q,D2)
        return out, QK
    
    
    #splits (B,L,dim) to (B, num_heads, L, head_dim)
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.heads, self.head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    
    def call(self,
             query,  #(B, Q, D1)
             keys,   #(B, K, D1)
             values, #(B, K, D2)
             mask=None):  #(B, Q, KV) (or broadcastable to that shape)
        
        batch_size = tf.shape(query)[0]
        
        #linear transform such that last dimensions fit
        query = self.wq(query)  
        keys = self.wk(keys) 
        values = self.wv(values) 
        
        #instead of a single attention over the whole model dimension, we
        #split dimensions to multiple heads to allow for different attention contexts
        query = self.split_heads(query, batch_size)  
        keys = self.split_heads(keys, batch_size) 
        values = self.split_heads(values, batch_size) 
        
        out, A = self.scaled_dot_product_attention(query, keys, values, mask)
        
        #transpose back and reshape to original form
        out = tf.transpose(out, perm=[0, 2, 1, 3])  
        out = tf.reshape(out, (batch_size, -1, self.dim))  
        return self.dense(out), A


##################################################################################################
##################################################################################################

#
# Encodes the input sequences by replacing each amino acid with it's
# high dimensional embedding. 
# Inputs are one-hot encoded sequences. Using a matrix multiplication with the embedding matrix 
# allows for modeling residual level uncertainty. For instance, if a sequence is only distantly related to others it might be
# highly mutated. However, these mutations are not completely random an favor few other amino acids given an original amino acid.
#
# The embedding is followed by 2 (unidirectional) LSTM layers which replace the conventional positional encoding used for 
# transformers. The LSTMs can also learn other important residual level features like secondary structure.
#
class EmbeddingAndPositionalEncoding(layers.Layer):
    def __init__(self, 
                 input_dim, 
                 embedding_dim, 
                 num_lstm, dropout, 
                 use_bidirectional_lstm):
        
        super(EmbeddingAndPositionalEncoding, self).__init__()
        
        self.matrix = self.add_weight(shape=(input_dim, embedding_dim),
                                        name='embedding', initializer="uniform", trainable=True)
        self.masking = layers.Masking()
        self.lstm = []
        for i in range(num_lstm):
            if use_bidirectional_lstm:
                forward_lstm = layers.LSTM(embedding_dim, return_sequences=True)
                backward_lstm = layers.LSTM(embedding_dim, return_sequences=True, go_backwards=True)
                self.lstm.append(layers.Bidirectional(
                    forward_lstm, 
                    backward_layer=backward_lstm, 
                    merge_mode="sum"))
            else:
                self.lstm.append(layers.LSTM(embedding_dim, return_sequences=True))
        self.dropout = layers.Dropout(dropout)
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)
    
    # x is a (num_seq, len_seq, D) tensor where D is assumed to be a probability distribution over the underlying alphabet
    # if use_gap_dummy == true, a shared gap dummy embedding is appended at the front of each sequence after positional encoding 
    # (i.e. the gap is positionless, it functions as a dummy to allow columns to attent to something in the sequence, if
    # to no amino acid)
    def call(self, x, training, use_gap_dummy):
        
        #embedding
        x_emb = tf.matmul(x, self.matrix)
       #x_emb *= tf.math.sqrt(tf.cast(tf.shape(x)[-1], tf.float32)) #not sure why they do this in the Transformer paper #XXXXX
        #I think this is only relevant when sharing the embedding matrix with output
        
        #encoding
        x_lstm = self.masking(x_emb)
        for lstm in self.lstm:
            x_lstm = lstm(x_lstm)
        x_lstm = self.dropout(x_lstm, training=training)
        x_lstm = self.layernorm(x_lstm + x_emb)
        if use_gap_dummy:
            seq_num = tf.shape(x_lstm)[0]
            gap_dummy = tf.reshape(self.matrix[data.GAP_MARKER], (1,1,-1))
            x_lstm = tf.concat([tf.repeat(gap_dummy, seq_num, axis=0), x_lstm], axis=1)
        return x_lstm

##################################################################################################
##################################################################################################

# 1x self attention + feedforward
# applies layernorm and dropout between these two and at the end
class EncoderLayer(layers.Layer):
    def __init__(self, dim, heads, dim_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(dim, heads)
        self.ffn = keras.Sequential( [layers.Dense(dim_ff, activation="relu"), layers.Dense(dim),] )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

    #seq.shape = (B, L, D)
    #mask.shape = (B, L, L) where 0 indicates a padded position pair
    def call(self, seq, mask, training):
        seq_attn, _ = self.self_attention(seq, seq, seq, mask)
        seq_attn = self.dropout1(seq_attn, training=training)
        seq_norm1 = self.layernorm1(seq + seq_attn)
        seq_ffn = self.ffn(seq_norm1)
        seq_ffn = self.dropout2(seq_ffn, training=training)
        return self.layernorm2(seq_norm1 + seq_ffn)


# a stack of encoder layers
class Encoder(tf.keras.layers.Layer):
    def __init__(self, iterations, dim, heads, dim_ff, dropout=0.1):
        super(Encoder, self).__init__()

        self.dim = dim
        self.enc_layers = [EncoderLayer(dim, heads, dim_ff, dropout) 
                           for _ in range(iterations)]

    def call(self, x, training, mask):
        for layer in self.enc_layers:
            x = layer(x, mask, training)
        return x  
    
    
    
    
###################################################################################################    
################################################################################################### 
    
    # cross aligns all input sequences to each other
class CrossTransformerLayer(layers.Layer):
    
    def __init__(self,
                 dim, 
                 heads, 
                 dim_ff, 
                 dropout=0.1):
        super(CrossTransformerLayer, self).__init__()
        
        self.dim = dim
        self.heads = heads
        self.cross_attention = MultiHeadAttention(dim, heads)
        self.inter_sequence_attention = MultiHeadAttention(dim, heads)
        self.ffn = keras.Sequential( [layers.Dense(dim_ff, activation="relu"), layers.Dense(dim),] )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)
    

    def call(self, 
             S, #(num_seq, L1, D1) 
             padding_mask, #(num_seq, 1, 1, L1)
             training): 
        
        num_seq = tf.shape(S)[0]
        len_seq = tf.shape(S)[1]
        
        # tile as follows: 
        # given sequences (s1, .., sk) 
        # queries: ( s1 x k, s2 x k, ...)
        # keys=values: ((s1,s2,s3,..,sk), (s1,s2,s3,..,sk),...)
        queries = tf.repeat(S, axis=0, repeats=tf.ones(num_seq, dtype=tf.int32)*num_seq)
        keyvalue = tf.tile(S, [num_seq, 1, 1])
        #padding_mask = tf.reshape(padding_mask, (1,1,len_seq,len_seq))
        #tiled_mask = tf.tile(padding_mask, [num_seq, 1, 1, 1])
        
        #cross attention between all sequence pairs (including self attentions)
        S_attn, _ = self.cross_attention(queries, keyvalue, keyvalue, padding_mask)
        
        #aggregate (reverse tiling)
        S_attn = tf.reshape(S_attn, (num_seq, num_seq, len_seq, self.dim))
        S_attn = tf.transpose(S_attn, perm=[1,2,0,3])
        S_attn = tf.reshape(S_attn, (-1, num_seq, self.dim))
        S_flip = tf.reshape(S, (-1, 1, self.dim))
        S_attn, _ = self.inter_sequence_attention(S_flip, S_attn, S_attn)
        S_attn = tf.reshape(S_attn, (num_seq, len_seq, self.dim))
        S_attn = self.dropout1(S_attn, training=training)
        S_norm = self.layernorm1(S + S_attn) 
        
        #feedforward, norm and skip connect
        S_ffn = self.ffn(S_norm)
        S_ffn = self.dropout2(S_ffn, training=training)
        return self.layernorm2(S_norm + S_ffn)
        

#given n sequences, return k transformed output sequences
class LatentEncoder(layers.Layer):
    
    def __init__(self,
                 num_latent_seq,
                 dim, 
                 heads, 
                 dim_ff, 
                 dropout=0.1):
        super(LatentEncoder, self).__init__()
        
        self.num_latent_seq = num_latent_seq
        
        self.latent_variables = self.add_weight(shape=(input_dim, embedding_dim),
                                        name='embedding', initializer="uniform", trainable=True)
        
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(dropout)
        
    def call(self, 
             S, #(num_seq, L, D) 
             padding_mask, #(num_seq, 1, 1, L)
             training): 
        pass

        
# n input sequences are first be transformed to k latent sequences, where k << n and constant.
# The k latent sequences are then cross transformed.
# Eventually, the k latent sequences are transformed back to n input sequences.
class CrossTransformer(layers.Layer):
    
    def __init__(self, iterations, dim, heads, dim_ff, dropout=0.1):
        super(CrossTransformer, self).__init__()
        
        self.cross_layers = [CrossTransformerLayer(dim, heads, dim_ff, dropout) 
                           for _ in range(iterations)]
        

    def call(self, S, training, padding_mask):
        
        for layer in self.cross_layers:
            S = layer(S, padding_mask,training)
        return S
    

##################################################################################################
##################################################################################################

# 1) self attention of the columns
# 2) attention between sequences as keys,values and columns as query
#    i.e. each output column will be a weighted average of all residues plus the gap dummy
# 3) feedforward 
# applies layernorm and dropout between all these steps and at the end
class DecoderLayer(layers.Layer):
    
    def __init__(self, dim, heads, dim_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.dim = dim
        self.heads = heads
        self.target_attention = MultiHeadAttention(dim, heads)
        self.input_to_target_attention = MultiHeadAttention(dim, heads) 
        #self.inter_sequence_attention = MultiHeadAttention(dim, heads)
        self.ffn1 = keras.Sequential( [layers.Dense(dim_ff, activation="relu"), layers.Dense(dim),] )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        #self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm4 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)
        #self.dropout3 = layers.Dropout(dropout)
        self.dropout4 = layers.Dropout(dropout)
        
       # self.ffn_message = keras.Sequential( [layers.Dense(dim_ff, activation="relu"), layers.Dense(dim),] )        #
        #self.ffn_update = keras.Sequential( [layers.Dense(dim_ff, activation="relu"), layers.Dense(dim),] )
        #self.dense = layers.Dense(dim)

    
    # Updates C via message passing and aggregations to
    # allow intersequencial and global information flow
    #def message_and_aggregate(self, X, training):
        
        #compute a message per attention head
        #message = self.ffn_message(X)
        #message = tf.reduce_sum(message, axis=0, keepdims=True) - message # does not change shape
       
        #X = tf.transpose(X, perm=[0, 2, 1, 3])  
        #X = tf.reshape(X, (tf.shape(X)[0], tf.shape(X)[1], self.dim))  
        #X = self.dense(X)
        #X = tf.expand_dims(X, 1)
        #X = tf.repeat(X, axis=1, repeats=self.heads)
        
        #C_upd = self.ffn_update(tf.concat([X, message], axis=-1))
        #C_upd = tf.reduce_mean(C_upd, axis=1)
        #return C_upd
    

    def call(self, 
             x, #(B, L1, D1)
             enc_input, #(B, L2, D2)
             look_ahead_mask, #broadcastable to (.., L2, L2) 
             padding_mask, #broadcastable to (.., L1, L2)
             training): 

        #target self attention, norm and skip connect
        x_attn1, _ = self.target_attention(x, x, x, look_ahead_mask)
        x_attn1 = self.dropout1(x_attn1, training=training)
        x_norm1 = self.layernorm1(x + x_attn1)

        #input to targets attention, norm and skip connect
        x_attn2, A = self.input_to_target_attention(x_norm1, enc_input, enc_input, padding_mask)
        x_attn2 = self.dropout2(x_attn2, training=training)
        x_norm2 = self.layernorm2(x_norm1 + x_attn2)
        
        #inter sequence attention
        #x_flip = tf.transpose(x_norm2, perm=[1,0,2])
        #x_attn3, _ = self.inter_sequence_attention(x_flip, x_flip, x_flip)
        #x_attn3 = tf.transpose(x_attn3, perm=[1,0,2])
        #x_norm3 = self.dropout3(x_attn3, training=training)
        #x_norm3 = self.layernorm3(x_norm2 + x_attn3)
        x_norm3 = x_norm2
        
        
        #feedforward, norm and skip connect
        x_ffn1 = self.ffn1(x_norm3)
        x_ffn1 = self.dropout4(x_ffn1, training=training)
        x_norm3 = self.layernorm4(x_norm3 + x_ffn1)
        
        return x_norm3, A
        
    

# a stack of decoder layers 
# "aggregation_iterations" : after each of these the sequence_aggregation function is called 
# to reduce the first dimension (=number of sequences)
# "iterations" : successive decoder layers without aggregation
class ColDecoder(layers.Layer):
    
    def __init__(self, iterations, dim, heads, dim_ff, sequence_aggregation, dropout=0.1):
        super(ColDecoder, self).__init__()
        
        self.dec_layers = [DecoderLayer(dim, heads, dim_ff, dropout) 
                           for _ in range(iterations)]
        

    def call(self, C, S, training, 
           look_ahead_mask, padding_mask):
        
        for layer in self.dec_layers:
            C, A = layer(C, S, look_ahead_mask, padding_mask,training)
        return C, A

##################################################################################################
##################################################################################################

class NeuroAlignLayer(layers.Layer):
    
    def __init__(self, in_dim, config):
        super(NeuroAlignLayer, self).__init__()
        
        self.config = config
        self.embedding = EmbeddingAndPositionalEncoding(in_dim, 
                                                        config["seq_dim"], 
                                                        config["num_lstm"], 
                                                        config["dropout"], 
                                                        use_bidirectional_lstm=False)
        #self.tar_embedding = EmbeddingAndPositionalEncoding(4, 
        #                                                    config["seq_dim"], 
        #                                                    config["num_lstm"], 
        #                                                    config["dropout"], 
        #                                                    use_bidirectional_lstm=False,
        #                                                    use_gap_dummy=False)
        self.encoder = Encoder(config["num_encoder_iterations"], 
                               config["seq_dim"], 
                               config["num_heads"], 
                               config["dim_ff"], 
                               config["dropout"])
        
        #self.cross_transfo = CrossTransformer(config["num_decoder_iterations"], 
        #                       config["seq_dim"], 
        #                       config["num_heads"], 
        #                       config["dim_ff"], 
        #                       config["dropout"])
            
        self.col_decoder = ColDecoder(config["num_decoder_iterations"], 
                                      config["col_dim"],
                                      config["num_heads"], 
                                      config["dim_ff"], 
                                      config["sequence_aggregation"], 
                                      config["dropout"])
        self.dense = layers.Dense(4, activation="softmax")
        


    def call(self, inp, tar, seq_padding_mask, col_look_ahead_mask, seq_decode_mask, training):
        
        num_seq = tf.shape(inp)[0]
        
        enc_inp = self.embedding(inp, training=training, use_gap_dummy=True)  
        self_inp = self.encoder(enc_inp, training, seq_padding_mask) 
        
        enc_tar = self.embedding(tar, training=training, use_gap_dummy=False) 
        
        out_cols, out_attention = self.col_decoder(enc_tar, self_inp, training, col_look_ahead_mask, seq_decode_mask)
        
        #out_cols = self.cross_transfo(out_cols, training, col_look_ahead_mask)
        #out_cols = tf.squeeze(out_cols, 0)
        
        # share the alphabet-embedding matrix for input/output
        # see paper: Using the output embedding to improve language models, Ofir Press and Lior Wolf, 2016
        #out_cols = tf.matmul(out_cols, self.tar_embedding.matrix, transpose_b = True)
        
        #out_cols = tf.nn.softmax(out_cols, axis=-1)
        
        out_cols = self.dense(out_cols)
        #out_cols = tf.squeeze(out_cols, -1)
        
        #out_attention = tf.squeeze(out_attention, 1)
        #out_attention = tf.transpose(out_attention, perm=[0,2,1])
        
        # If the softmax is over the columns, the model outputs a
        # probability distribution for each residuum.
        # However, this approach may not be correct when doing autoregressive inference
        # because a residuum might be undecided for its column if only the
        # first i columns are predicted yet.
        # Softmaxing over the sequences can therefore be a more natural choice because 
        # the introduced gap dummy symbol allows a column to decide not to attend to any
        # residuum at all.
        # However, the model might let 2 columns attent to the same residuum in this case.
        #if self.config["softmax_over_columns"]:
            #softmax_axis = -1
        #else:
            #softmax_axis = -2
            
         
        out_attention = tf.nn.softmax(out_attention[:,:-1,2:-1], axis=-2)
            
        return out_cols, out_attention
        
        
    
#####################################################################################################################################
#####################################################################################################################################

INPUT_DIM = len(data.ALPHABET) + 3
    
#create the model using keras functional api
def make_neuro_align_model(identifier):
    
    cfg = config.models[identifier]
    
    #inputs
    sequences = keras.Input(shape=(None,INPUT_DIM), name="sequences")
    aligned_sequences = keras.Input(shape=(None,INPUT_DIM), name="aligned_sequences") 
    sequences_residual_mask = keras.Input(shape=(None,None), name="sequences_residual_mask") 
    num_seq = tf.shape(sequences)[0]
    
    #layers
    NR = NeuroAlignLayer(INPUT_DIM, cfg)
    
    FF = layers.Dense(INPUT_DIM, activation="softmax") #+ start- and end-marker
    
    #masks
    #append ones to account for the gap dummy which is concatenated later
    seq_padding_mask = make_padding_mask(tf.concat([tf.ones((num_seq,1,tf.shape(sequences)[-1])),
                                        sequences], axis=1))
    look_ahead_mask = make_look_ahead_mask(aligned_sequences)
    sequences_residual_mask_ = tf.concat([tf.ones((num_seq,tf.shape(sequences_residual_mask)[-2],1)), sequences_residual_mask], axis=-1)
    sequences_residual_mask_ = tf.expand_dims(sequences_residual_mask_, 1)
    seq_decode_mask = tf.math.minimum(seq_padding_mask, sequences_residual_mask_)
    
    #compute output
    out_gaps, A = NR(sequences, aligned_sequences, seq_padding_mask, look_ahead_mask, seq_decode_mask)
    
    #instantiate the model
    outputs = []
    outputs.append(layers.Lambda(lambda x: x, name="out_gaps")(out_gaps)) #lambda identity used for renaming
    #outputs.append(layers.Lambda(lambda x: x, name="out_attention")(A))
    model = keras.Model(inputs=[sequences, aligned_sequences, sequences_residual_mask], outputs=outputs)
    
    checkpoint_path = "./models/"+identifier+"/model.ckpt"
    if os.path.isfile(checkpoint_path+".index"):
        model.load_weights(checkpoint_path)
        print("Successfully loaded weights:", identifier, "model")
    else:
        print("Configured model", identifier, "and initialized weights randomly.")

    return model, cfg


#####################################################################################################################################
#####################################################################################################################################

# Autoregressively generates a sequence of columns starting with only the "column start" marker.
# The output for the next column depends on all previously predicted columns and the evidence from the input sequences.
# The generation is unrolled until the model outputs a "column end" marker or the maximum sequence length is reached.
# If column loss is not enabled, no unrolling occurs. Instead, the output is based on a single iteration on "empty"
# columns with only positional encoding. The alignment length is naively estimated by a multiple of the longest sequence 
# length in the input.
def gen_columns(sequences, seq_lens, model, model_config, max_length=1000):
    
    num_seq = sequences.shape[0]
    len_seq = sequences.shape[1]
    aligned_sequences = np.zeros((num_seq, max_length+1, len(data.ALPHABET)+3))
    aligned_sequences[:,0, data.START_MARKER] = 1 #start marker
    sequences_residual_mask = np.ones((num_seq, max_length+1, len_seq))
    sequences_residual_mask[:, :, 0] = 0
    
    #A = np.zeros((tf.shape(sequences)[0], tf.shape(sequences)[1], max_length))
    #indices = np.ones(tf.shape(sequences)[0], dtype=int)
    
    for i in range(max_length):
        inp = {"sequences" : sequences, 
               "aligned_sequences" : aligned_sequences[:, :(i+1)],
               "sequences_residual_mask" : sequences_residual_mask[:, :(i+1), :]}
        print(sequences.shape)
        out_gaps = model(inp, training=False)
        print(out_gaps.shape)
        print(out_gaps[:,-1])
        advance = np.logical_and(out_gaps[:,-1] < 0.5, indices < seq_lens)
        print(advance)
        if np.all(np.logical_not(advance)):
            return A[:,:,:i]
        A[advance, indices[advance], i] = 1
        next_column = np.sum(sequences[advance, indices[advance]], axis=0) / np.sum(advance)
        indices += advance
        if np.all(indices >= seq_lens):
            return A[:,:,:(i+1)]
        else:
            columns[i+1] = next_column
    return A