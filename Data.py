import numpy as np
import copy
import random
import tensorflow as tf
from tensorflow import keras

##################################################################################################
##################################################################################################

#NeuroAlign can operate on any finite alphabet, here the extended aminoacid alphabet is hard-coded
#ALPHABET[:20] corresponds to the traditional aminoacid alphabet
ALPHABET = ['A', 'R',  'N',  'D',  'C',  'Q',  'E',  'G',  'H', 'I',  'L',  'K',  
            'M',  'F',  'P', 'S',  'T',  'W',  'Y',  'V',  'B',  'Z',  'X', 'U', 'O']


GAP_MARKER = len(ALPHABET)
START_MARKER = len(ALPHABET)+1
END_MARKER = len(ALPHABET)+2

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
        cumsum = np.cumsum(self.ref_seq != len(ALPHABET), axis=1) #A-B--C -> 112223
        diff = np.diff(np.insert(cumsum, 0, 0.0, axis=1), axis=1) #112223 -> 0112223 -> [[(i+1) - i]] -> 101001
        diff_where = [np.argwhere(diff[i,:]).flatten() for i in range(diff.shape[0])]
        self.membership_targets = np.concatenate(diff_where).flatten()
        
    
    
    #converts (a subset of) the sequences to one hot encodings, appends gap-, start- and end-markers
    #mode defines how the sequences are layouted 
    # leftbound writes them from left to write and fills short sequences with padded zeros
    # uniform multiplies the longest sequence length by a constant a little greater than one
    # and distributes the positions of each sequence uniformly at this space filling inbetweens with gaps
    def one_hot_sequences(self, subset=None, mode="leftbound"):
        if subset is None:
            subset = list(range(len(self.raw_seq)))
        num_seqs = len(subset)
        lens = [self.seq_lens[si] for si in subset]
        maxlen = max(lens)
        ranges = []
        if mode == "leftbound":
            seq = np.zeros((num_seqs, maxlen+2, len(ALPHABET)+3), dtype=np.float32)
            for j,(l,si) in enumerate(zip(lens, subset)):
                lrange = np.arange(l)+1
                seq[j, lrange, self.raw_seq[si]] = 1
                seq[j, 1+l, END_MARKER] = 1 #end marker
                ranges.append(lrange)
            #first position = gap symbol (more details later, will not be affected by positional encoding)
            #seq[:, 0, GAP_MARKER] = 1 #start marker
            seq[:, 0, START_MARKER] = 1
            return seq, maxlen+2, ranges, lens
        else: #mode == "uniform"
            L = int(np.floor(maxlen*1.1))+2
            seq = np.zeros((num_seqs, L, len(ALPHABET)+3), dtype=np.float32)
            ranges = []
            for j,(l,si) in enumerate(zip(lens, subset)):
                lrange = np.floor((np.arange(l)+1)*(L-1)/(l+1)).astype(int)
                indx = np.zeros(L, dtype=np.bool)
                indx[lrange] = 1
                seq[j, indx, self.raw_seq[si]] = 1
                seq[j, ~indx, GAP_MARKER] = 1
                ranges.append(lrange)
            seq[:, 0, START_MARKER] = 1
            seq[:, 0, GAP_MARKER] = 0
            seq[:, L-1, END_MARKER] = 1 #end marker
            seq[:, L-1, GAP_MARKER] = 0
            return seq, L, ranges, [L-2]*num_seqs
    
    
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
     
    seq, L, ranges, lens = family_fasta.one_hot_sequences(subset = seqs_drawn, mode=model_config["sequence_layout"])
    num_seqs = len(seqs_drawn)
    
    #remove empty columns 
    col_sizes = np.zeros(family_fasta.alignment_len)
    for j,(l, si) in enumerate(zip([family_fasta.raw_seq[i].shape[0] for i in seqs_drawn], seqs_drawn)):
        suml = sum(family_fasta.seq_lens[:si])
        col_sizes[family_fasta.membership_targets[suml:(suml+l)]] += 1
    empty = (col_sizes == 0)
    num_columns = int(np.sum(~empty))
    cum_cols = np.cumsum(empty)

    corrected_targets = []
    for j,(l, si) in enumerate(zip([family_fasta.raw_seq[i].shape[0] for i in seqs_drawn], seqs_drawn)):
        suml = sum(family_fasta.seq_lens[:si])
        ct = family_fasta.membership_targets[suml:(suml+l)]
        corrected_targets.append(ct - cum_cols[ct])
       
    if model_config["sequence_layout"] == "leftbound":
        base = 2
    else:
        base = 1
        
    memberships = np.zeros((num_seqs, L, num_columns+2), dtype=np.float32)
    for j,(l, lrange, targets) in enumerate(zip(lens, ranges, corrected_targets)):
        #memberships site <-> columns
        memberships[j, lrange, 1+targets] = 1
        #memberships[j, base+l, -1] = 1 
        
    #if model_config["sequence_layout"] == "leftbound":
        #memberships[:,1,0] = 1 
    #if model_config["pairs_with_gaps"]:
        #if model_config["sequence_layout"] == "leftbound":
            #memberships[:,0,:] = 1 - np.sum(memberships[:,1:,:], axis=1) #gaps
        #else:
            #memberships[:,0,:] = 1 - np.sum(memberships[:,1:,:], axis=1) #gaps
    #else:
        #memberships[:,0,0] = 1 

    #the matmul computes aminoacid count vectors for each alignment column
    #furthermore, we add a "start" and an "end" marker and compute the number of gaps per column
    columns = np.matmul(np.transpose(memberships, [0,2,1]), seq)
    columns = np.sum(columns, axis=0)
    columns /= num_seqs
    columns[1:-1, GAP_MARKER] = 1 - np.sum(columns[1:-1,:], axis=1)
    columns[0, START_MARKER] = 1 
    columns[-1, END_MARKER] = 1 
    if model_config["use_column_loss"]:
        in_columns = columns[:-1]
    else:
        in_columns = np.zeros_like(columns[:-1])
    out_columns = columns[1:]
        
    out_memberships = memberships[:,:,1:]
    
    
    input_dict = {  ext+"sequences" : seq,
                     ext+"in_columns" : in_columns }
        
    target_dict = { ext+"out_columns" : out_columns,
                      ext+"out_attention" : out_memberships }
        
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
            if len(self.fasta) > 10:
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