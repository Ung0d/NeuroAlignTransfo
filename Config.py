import tensorflow as tf

base_model = {
    
    #dimensionality of the latent representation for each residuum
    "seq_dim" : 128,
    
    #dimensionality of the latent representation for each column
    "col_dim" : 128,
    
    #dimensionality of all feed forward layers
    "dim_ff" : 256,
    
    #multi head attention; applies to all attention mechanisms except the last 
    #sequence-to-columns steps, where the number of heads is fixed to 1
    "num_heads" : 8,
    
    "single_head_seq_to_col" : True,
    
    #number of forward LSTMs used for positional encoding and learning other residual level features
    "num_lstm" : 2,
    
    #how many self attention layers for both sequences and columns
    "num_encoder_iterations" : 3,
    
    #how many self attention layers for the columns without an aggregation inbetween
    "num_decoder_iterations" : 1,
    
    #how many aggregations
    "num_aggregation_iterations" : 3,

    "dropout" : 0.1,
    
    "sequence_aggregation" : tf.reduce_mean,

    #when training, we feed one alignment per GPU at a time (since alignments are large)
    #the family size is an upper limit for the total sequence length in a sample
    "family_size" : 4000,

    #percentage of PFam protein families that become the validation set
    "validation_split" : 0.05
}

base_model2 = dict(base_model)
base_model2["num_encoder_iterations"] = 3
base_model2["num_aggregation_iterations"] = 3
base_model2["single_head_seq_to_col"] = False

dirichlet = dict(base_model)
dirichlet["num_encoder_iterations"] = 2
dirichlet["num_aggregation_iterations"] = 2
dirichlet["single_head_seq_to_col"] = False
dirichlet["seq_dim"] = 64
dirichlet["col_dim"] = 64
dirichlet["dim_ff"] = 128
dirichlet["num_heads"] = 4
dirichlet["num_lstm"] = 1

gap_prob = dict(dirichlet)
gap_prob["dropout"] = 0.1
gap_prob["sequence_aggregation"] = tf.reduce_sum


models = {"base" : base_model,
         "base2" : base_model2,
         "dirichlet" : dirichlet,
         "full_col_dist" : dirichlet,
         "gap_prob" : gap_prob}