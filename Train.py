import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import CSVLogger
import Model as model
import Data as data
import Evaluation as eval

GPUS = tf.config.experimental.list_logical_devices('GPU')
NUM_DEVICES = max(1, len(GPUS))

if len(GPUS) > 0:
    print("Using ", NUM_DEVICES, " GPU devices.")
else:
    print("Using CPU.")
    
    
    
NUM_EPOCHS = 200
NAME = "base"
MODEL_PATH = "./models/" + NAME
CHECKPOINT_PATH = MODEL_PATH + "/model.ckpt"

os.makedirs(MODEL_PATH, exist_ok=True)


##################################################################################################
##################################################################################################
neuroalign, neuroalign_config = model.make_neuro_align_model(NAME)

#Pfam protein families have identifiers of the form PF00001, PF00002, ...
#The largest id is PF19227, but the counting is not contiguous, there may be missing numbers
pfam = ["PF"+"{0:0=5d}".format(i) for i in range(1,19228)]
pfam_not_found = 0

fasta = []

for i,file in enumerate(pfam):
    try:
        f = data.Fasta("../Pfam/alignments/" + file + ".fasta", gaps = True, contains_lower_case = True)
        fasta.append(f)
        for x in range(1,10):
            if i/len(pfam) > x/10 and (i-1)/len(pfam) < x/10:
                print(x*10, "% loaded")
                gc.collect()
    except:
        pfam_not_found += 1

np.random.seed(0)
random.seed(0)

indices = np.arange(len(fasta))
np.random.shuffle(indices)
if len(fasta) > 10:
    print("Using the full dataset.")
    train, val = np.split(indices, [int(len(fasta)*(1-neuroalign_config["validation_split"]))]) 
    train_gen = data.AlignmentSampleGenerator(train, fasta, neuroalign_config, neuroalign_config["family_size"], NUM_DEVICES)
    val_gen = data.AlignmentSampleGenerator(val, fasta, neuroalign_config, 2*neuroalign_config["family_size"], NUM_DEVICES, False)
else: 
    print("Using a small test dataset.")
    train_gen = data.AlignmentSampleGenerator(np.arange(len(fasta)), fasta, neuroalign_config, neuroalign_config["family_size"], NUM_DEVICES)
    val_gen = data.AlignmentSampleGenerator(np.arange(len(fasta)), fasta, neuroalign_config, 2*neuroalign_config["family_size"], NUM_DEVICES, False) 
    
    
    
    
INPUT_DIM = 27

#COLUMN_LOSS_WEIGHT = 0.02
#ATTENTION_LOSS_WEIGHT = 0.98

##################################################################################################
##################################################################################################

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = tf.cast(neuroalign_config["col_dim"], tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


##################################################################################################
##################################################################################################

optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.9, beta_2=0.98, 
                                     epsilon=1e-9)

##################################################################################################
##################################################################################################


def losses_prefixed(losses, metrics, weights, prefix=""):
    #if neuroalign_config["use_column_loss"]:
        #losses.update({prefix+"out_columns" : eval.kld})
        #weights.update({prefix+"out_columns" : COLUMN_LOSS_WEIGHT})
    #if neuroalign_config["use_attention_loss"]:
        #losses.update({prefix+"out_attention" : eval.att_loss})
        #metrics.update({prefix+"out_attention" : [eval.precision, eval.recall]})
        #weights.update({prefix+"out_attention" : ATTENTION_LOSS_WEIGHT})
    losses.update({prefix+"out_gaps" : 
                   keras.losses.CategoricalCrossentropy(
                       label_smoothing=0.1)})
    metrics.update({prefix+"out_gaps" : keras.metrics.CategoricalAccuracy()})
        

losses, metrics, weights = {}, {}, {}
if NUM_DEVICES == 1:
    model = neuroalign
    losses_prefixed(losses, metrics, weights)
else:
    inputs, outputs = [], []
    for i, gpu in enumerate(GPUS):
        with tf.device(gpu.name):
            sequences = keras.Input(shape=(None,INPUT_DIM), name="GPU_"+str(i)+"_sequences")
            in_gaps = keras.Input(shape=(None,4), name="GPU_"+str(i)+"_in_gaps")
            input_dict = {  "sequences" : sequences,
                            "in_gaps" : in_gaps }
            #out_cols, A = neuroalign(input_dict)
            out_gaps = neuroalign(input_dict)
            outputs.append(layers.Lambda(lambda x: x, name="GPU_"+str(i)+"_out_gaps")(out_gaps))
            #outputs.append(layers.Lambda(lambda x: x, name="GPU_"+str(i)+"_out_attention")(A))
            inputs.extend([sequences, in_gaps])

    model = keras.Model(inputs=inputs, outputs=outputs)
    for i, gpu in enumerate(GPUS):
        losses_prefixed(losses, metrics, weights, "GPU_"+str(i)+"_")

model.compile(loss=losses, optimizer=optimizer, metrics=metrics, loss_weights=weights)
    
class ModelCheckpoint(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        neuroalign.save_weights(CHECKPOINT_PATH)
        print("Saved model to " + CHECKPOINT_PATH, flush=True)

csv_logger = CSVLogger(MODEL_PATH + "/log.csv", append=True, separator=',')

history = model.fit(train_gen,
                    validation_data=val_gen,
                    epochs = NUM_EPOCHS,
                    verbose = 2,
                    callbacks=[ModelCheckpoint(), csv_logger])
