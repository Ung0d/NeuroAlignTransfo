import tensorflow as tf

base_model = {
    
    #train the aminoacid distributions of the output columns
    "use_column_loss" : True,
    
    #train the probability of aligning a aminoacid pair (=squared attention weight matrix) 
    "use_attention_loss" : True,
    
    #if true, the aminoacid pair loss also considers gaps 
    "pairs_with_gaps" : False,
    
    #if true, the attention scores are softmaxed over the columns in the output
    #soft alignment, if false, the softmax is over the sequences plus a dummy gap symbol (positionless)
    "softmax_over_columns" : True,
    
    #dimensionality of the latent representation for each residuum
    "seq_dim" : 128,
    
    #dimensionality of the latent representation for each column
    "col_dim" : 128,
    
    #dimensionality of all feed forward layers
    "dim_ff" : 256,
    
    #multi head attention; applies to all attention mechanisms except the last 
    #sequence-to-columns steps, where the number of heads is fixed to 1
    "num_heads" : 4,
    
    "single_head_seq_to_col" : False,
    
    #how many self attention layers for both sequences and columns
    "num_encoder_iterations" : 2,
    
    #how many self attention layers for the columns without an aggregation inbetween
    "num_decoder_iterations" : 1,
    
    #how many aggregations
    "num_aggregation_iterations" : 2,

    "dropout" : 0.05,
    
    "sequence_aggregation" : tf.reduce_mean,

    #when training, we feed one alignment per GPU at a time (since alignments are large)
    #the family size is an upper limit for the total sequence length in a sample
    "family_size" : 4000,

    #percentage of PFam protein families that become the validation set
    "validation_split" : 0.05,
    
    "sequence_layout" : "leftbound"
}


base_model2 = {
    
    #train the aminoacid distributions of the output columns
    "use_column_loss" : True,
    
    #train the probability of aligning a aminoacid pair (=squared attention weight matrix) 
    "use_attention_loss" : True,
    
    #if true, the aminoacid pair loss also considers gaps 
    "pairs_with_gaps" : True,
    
    #if true, the attention scores are softmaxed over the columns in the output
    #soft alignment, if false, the softmax is over the sequences plus a dummy gap symbol (positionless)
    "softmax_over_columns" : False,
    
    #dimensionality of the latent representation for each residuum
    "seq_dim" : 128,
    
    #dimensionality of the latent representation for each column
    "col_dim" : 128,
    
    #dimensionality of all feed forward layers
    "dim_ff" : 256,
    
    #multi head attention; applies to all attention mechanisms except the last 
    #sequence-to-columns steps, where the number of heads is fixed to 1
    "num_heads" : 8,
    
    "single_head_seq_to_col" : False,
    
    #how many self attention layers for both sequences and columns
    "num_encoder_iterations" : 2,
    
    #how many self attention layers for the columns without an aggregation inbetween
    "num_decoder_iterations" : 1,
    
    #how many aggregations
    "num_aggregation_iterations" : 2,

    "dropout" : 0.1,
    
    "sequence_aggregation" : tf.reduce_mean,

    #when training, we feed one alignment per GPU at a time (since alignments are large)
    #the family size is an upper limit for the total sequence length in a sample
    "family_size" : 4000,

    #percentage of PFam protein families that become the validation set
    "validation_split" : 0.05,
    
    "sequence_layout" : "leftbound"
}



base_16head_pairs_with_gaps ={
    
    #train the aminoacid distributions of the output columns
    "use_column_loss" : True,
    
    #train the probability of aligning a aminoacid pair (=squared attention weight matrix) 
    "use_attention_loss" : True,
    
    #if true, the aminoacid pair loss also considers gaps 
    "pairs_with_gaps" : True,
    
    #if true, the attention scores are softmaxed over the columns in the output
    #soft alignment, if false, the softmax is over the sequences plus a dummy gap symbol (positionless)
    "softmax_over_columns" : True,
    
    #dimensionality of the latent representation for each residuum
    "seq_dim" : 128,
    
    #dimensionality of the latent representation for each column
    "col_dim" : 128,
    
    #dimensionality of all feed forward layers
    "dim_ff" : 256,
    
    #multi head attention; applies to all attention mechanisms except the last 
    #sequence-to-columns steps, where the number of heads is fixed to 1
    "num_heads" : 16,
    
    "single_head_seq_to_col" : True,
    
    #how many self attention layers for both sequences and columns
    "num_encoder_iterations" : 2,
    
    #how many self attention layers for the columns without an aggregation inbetween
    "num_decoder_iterations" : 1,
    
    #how many aggregations
    "num_aggregation_iterations" : 2,

    "dropout" : 0.1,
    
    "sequence_aggregation" : tf.reduce_mean,

    #when training, we feed one alignment per GPU at a time (since alignments are large)
    #the family size is an upper limit for the total sequence length in a sample
    "family_size" : 4000,
    
    "sequence_layout" : "leftbound"
}


uniform_seqs = dict(base_model)
uniform_seqs["num_heads"] = 4
uniform_seqs["num_encoder_iterations"] = 2
uniform_seqs["num_aggregation_iterations"] = 2
#uniform_seqs["sequence_layout"] = "uniform"


models = {"base" : base_model,
          "base2" : base_model2,
          "uniform_seqs" : uniform_seqs,
          "uniform_seqs2" : uniform_seqs,
          "test" : uniform_seqs,
         "base_16head_pairs_with_gaps" : base_16head_pairs_with_gaps}