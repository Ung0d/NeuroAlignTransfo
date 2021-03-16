## NeuroAlignTransfo

NeuroAlign is a deep learning based end-to-end model that is able to construct multiple sequence alignments by learning from datasets of reference alignments. Our model maps raw sequences
\- for instance over the standard aminoacid alphabet but not limited to that \- to a soft alignment represented by membership probabilities for each residuum to each alignment column out of (dynamically sized) consensus sequence. NeuroAlign can also output representation vectors for the alignment columns which encode the respective distribution of aminoacids that can be useful for higher level learning tasks.

We provide weight files to load configured and pretained models out of the box. Out models are trained on the Pfam database http://pfam.xfam.org/. 

### Requirements

NeuroAlign requires tensorflow (2.1 or higher) and for the Inference notebook matplotlib. Other than that, the repository is self-contained. 

If a GPU is present, the notebooks will automatically attempt to use it. The base model (./models/base) requires at least 7GBs of GPU memory. 
A CPU run can be forced by setting a flag.
