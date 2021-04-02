import numpy as np
import copy
import random
import tensorflow as tf
from tensorflow import keras
import DirichletMixturePrior as dirichlet

##################################################################################################
##################################################################################################

#NeuroAlign can operate on any finite alphabet, here the extended aminoacid alphabet is hard-coded
#ALPHABET[:20] corresponds to the traditional aminoacid alphabet
ALPHABET = ['A', 'R',  'N',  'D',  'C',  'Q',  'E',  'G',  'H', 'I',  'L',  'K',  
            'M',  'F',  'P', 'S',  'T',  'W',  'Y',  'V',  'B',  'Z',  'X', 'U', 'O']


START_MARKER = len(ALPHABET)
END_MARKER = len(ALPHABET)+1

##################################################################################################
##################################################################################################

dirichlet_mix_prior = dirichlet.make_prior(ALPHABET)

##################################################################################################
##################################################################################################

# a class that reads sequences in fasta file format and converts to 
# inputs and targets interpretable by the NeuroAlign model
class Fasta:
    def __init__(self, 
                 filename, #fasta file to parse
                 gaps = True, #does the file contain gaps or just raw sequences?
                 contains_lower_case = False #expect lower case AA encodings?
                ):
        self.filename = filename
        self.seq_ids = []
        self.valid = self.read_seqs(filename, gaps, contains_lower_case)
        if self.valid and gaps:
            self.compute_targets()

    def read_seqs(self, filename, gaps, contains_lower_case):

        #read seqs as strings
        with open(filename) as f:
            content = f.readlines()
        self.raw_seq = []
        seq_open = False
        for line in content:
            line = line.strip()
            if len(line)>0:
                if line[0]=='>':
                    seq_open = True
                    self.seq_ids.append(line[1:])
                elif seq_open:
                    self.raw_seq.append(line)
                    seq_open = False
                else:
                    self.raw_seq[-1] += line

        #ignore files containing "/" 
        for seq in self.raw_seq:
            if not seq.find("/") == -1:
                return False

        if gaps:
            self.alignment_len = len(self.raw_seq[0])

        self.raw_seq = [s.replace('.','-') for s in self.raw_seq] #treat dots as gaps
        for i,c in enumerate(ALPHABET):
            self.raw_seq = [s.replace(c,str(i)+' ') for s in self.raw_seq]
            if contains_lower_case:
                self.raw_seq = [s.replace(c.lower(),str(i)+' ') for s in self.raw_seq]

        #can store sequences with gaps as matrix
        if gaps:
            self.ref_seq = copy.deepcopy(self.raw_seq)
            self.ref_seq = [s.replace('-',str(len(ALPHABET))+' ') for s in self.ref_seq]
            self.ref_seq = np.reshape(np.fromstring("".join(self.ref_seq), dtype=int, sep=' '), (len(self.ref_seq), self.alignment_len))

        self.raw_seq = [s.replace('-','') for s in self.raw_seq]
        self.raw_seq = [np.fromstring(s, dtype=int, sep=' ') for s in self.raw_seq]
        self.seq_lens = [s.shape[0] for s in self.raw_seq]
        self.total_len = sum(self.seq_lens)
        return True



    def compute_targets(self):
        
        #a mapping from raw position to column index
        #A-B--C -> 112223
        cumsum = np.cumsum(self.ref_seq != len(ALPHABET), axis=1) 
        #112223 -> 0112223 -> [[(i+1) - i]] -> 101001
        diff = np.diff(np.insert(cumsum, 0, 0.0, axis=1), axis=1) 
        diff_where = [np.argwhere(diff[i,:]).flatten() for i in range(diff.shape[0])]
        self.membership_targets = np.concatenate(diff_where).flatten()
        
        seq = self.one_hot_sequences(append_markers=False)
        memberships, _ = self.column_memberships()
        
        #the matmul computes aminoacid count vectors for each alignment column
        column_count_vectors = np.matmul(memberships, seq)
        self.column_count_vectors = np.sum(column_count_vectors, axis=0)
        #posterior distributions for all columns
        #posterior_component_prob = dirichlet_mix_prior.posterior_component_prob(
                                                            #self.column_count_vectors[:,:20])
        #self.columns = dirichlet_mix_prior.posterior_amino_acid_prob(
                                                            #self.column_count_vectors[:,:20], 
                                                            #posterior_component_prob).numpy()
        self.columns = self.column_count_vectors / np.sum(self.column_count_vectors, keepdims=True, axis=-1)
        
    
    
    #converts (a subset of) the sequences to one hot encodings, optionally appends gap-, start- and end-markers
    def one_hot_sequences(self, append_markers=True, subset=None):
        if subset is None:
            subset = list(range(len(self.raw_seq)))
        num_seqs = len(subset)
        lens = [self.seq_lens[si] for si in subset]
        maxlen = max(lens)
        alphabet_size = len(ALPHABET)
        if append_markers:
            maxlen += 2
            alphabet_size += 2
        seq = np.zeros((num_seqs, maxlen, alphabet_size), dtype=np.float32)
        for j,(l,si) in enumerate(zip(lens, subset)):
            lrange = np.arange(l)
            if append_markers:
                seq[j, 1+lrange, self.raw_seq[si]] = 1
                seq[j, 1+l, END_MARKER] = 1 #end marker
            else:
                seq[j, lrange, self.raw_seq[si]] = 1
        if append_markers:
            seq[:, 0, START_MARKER] = 1
        return seq
    
    
    #returns a binary matrix indicating which residues correspond to which columns
    #if only a subset of the sequences is required, empty columns will be removed
    #output shape is (num_seq, num_columns, max_len_seq)
    def column_memberships(self, subset=None):
        if subset is None:
            subset = list(range(len(self.raw_seq)))
        
        lens = [self.seq_lens[si] for si in subset]
        maxlen = max(lens)
        num_seqs = len(subset)
        
        #remove empty columns 
        col_sizes = np.zeros(self.alignment_len)
        for j,(l, si) in enumerate(zip(lens, subset)):
            suml = sum(self.seq_lens[:si])
            col_sizes[self.membership_targets[suml:(suml+l)]] += 1
        empty = (col_sizes == 0)
        num_columns = int(np.sum(~empty))
        cum_cols = np.cumsum(empty)
            
        #shift column indices according to removed empty columns
        corrected_targets = []
        target_subset = np.zeros(self.alignment_len, dtype=bool)
        for j,(l, si) in enumerate(zip(lens, subset)):
            suml = sum(self.seq_lens[:si])
            ct = self.membership_targets[suml:(suml+l)]
            target_subset[ct] = 1
            corrected_targets.append(ct - cum_cols[ct])

        memberships = np.zeros((num_seqs, maxlen, num_columns), dtype=np.float32)
        for j,(l, targets) in enumerate(zip(lens, corrected_targets)):
            lrange = np.arange(l)
            memberships[j, lrange, targets] = 1
            
            
        return np.transpose(memberships, [0,2,1]), target_subset
            
        
    
    
    def aminoacid_seq_str(self, i):
        seq = ""
        for j in self.raw_seq[i]:
            seq += ALPHABET[j]
        return seq
    
    
    def column_str(self, i):
        col = ""
        ALPHABET_with_gap = ALPHABET + ["-"]
        for j in self.ref_seq[:,i]:
            col += ALPHABET_with_gap[j]
        return col


##################################################################################################
##################################################################################################


def get_input_target_data(family_fasta, seqs_drawn, 
                          model_config,
                          ext = ""):
     
    seq = family_fasta.one_hot_sequences(subset = seqs_drawn)
    memberships, target_subset = family_fasta.column_memberships(subset = seqs_drawn)
    
    columns = np.zeros((memberships.shape[1]+2, len(ALPHABET)+2))
    columns[1:-1, :len(ALPHABET)] = family_fasta.columns[target_subset]
    columns[0, START_MARKER] = 1
    columns[-1, END_MARKER] = 1
    in_columns = columns[:-1]
    out_columns = columns[1:]
    
    gaps = np.zeros((tf.shape(memberships)[0], tf.shape(memberships)[1]+2, 4))
    gaps[:, 1:-1, 0] = 1 - np.sum(memberships, axis=2)
    gaps[:, 1:-1, 1] = np.sum(memberships, axis=2)
    gaps[:, 0, 2] = 1 #start marker
    gaps[:, -1, 3] = 1 #end marker
    in_gaps = gaps[:,:-1,:]
    out_gaps = gaps[:,1:,:]
    
    input_dict = {  ext+"sequences" : seq,
                    ext+"in_gaps" : in_gaps }
        
    target_dict = { #ext+"out_columns" : out_columns,
                    #ext+"out_attention" : memberships,
                    ext+"out_gaps" : out_gaps }
        
    return input_dict, target_dict


##################################################################################################
##################################################################################################

#percentage of PFam protein families shown to the model during one epoch
EPOCH_PART = 1


#generates a sample of sequences from a randomly drawn protein family
#each sample has an upper limit of sites
#sequences are drawn randomly from the family until all available sequences are chosen or
#the family limit is exhausted.
class AlignmentSampleGenerator(keras.utils.Sequence):

    def __init__(self, split, fasta, model_config, family_size, num_devices, training = True):
        self.split = split
        self.model_config = model_config
        self.family_size = family_size
        self.num_devices = num_devices
        self.fasta = fasta
        #weights for random draw are such that large alignments that to not fit into a single batch
        #are drawn more often
        family_lens_total = [sum(fasta[i].seq_lens) for i in self.split]
        family_weights = [max(1, t/family_size) for t in family_lens_total]
        sum_w = sum(family_weights)
        self.family_probs = [w/sum_w for w in family_weights]
        self.training = training

    def __len__(self):
        if self.training:
            if len(self.fasta) > 1000:
                return int(len(self.split)*EPOCH_PART)
            else: 
                return 1000
        else:
            return len(self.split)

    def sample_family(self, ext = ""):

        #draw a random family and sequences from that family
        family_i = np.random.choice(len(self.split), p=self.family_probs)
        family_fasta = self.fasta[self.split[family_i]]
        
        seq_shuffled = list(range(len(family_fasta.raw_seq)))
        random.shuffle(seq_shuffled)
        family_size = 0
        seqs_drawn = []
        for si in seq_shuffled:
            if family_fasta.seq_lens[si] > 0:
                family_size += family_fasta.seq_lens[si]
                if family_size > self.family_size:
                    break
                else:
                    seqs_drawn.append(si)
        if len(seqs_drawn) < 2:
            return self.sample_family(ext)
        
        return get_input_target_data(family_fasta, seqs_drawn, self.model_config, ext)


    def __getitem__(self, index):
        if self.num_devices == 1:
            id, td, _,_,_ = self.sample_family()
            return id, td
        else:
            input_dict, target_dict = {}, {}
            for i in range(self.num_devices):
                id, td = self.sample_family("GPU_"+str(i)+"_")
                input_dict.update(id)
                target_dict.update(td),
            return input_dict, target_dict