# SLIM_RAA

Python 3.8.3 and Pytorch 1.12.1 implementation of the latent Signed relational Latent dIstance Model (SLIM).

## Description

Graph representation learning has become a prominent tool for the characterization and understanding of the structure of networks in general and social networks in particular. Typically, these representation learning approaches embed the networks into a low-dimensional space in which the role of each individual can be characterized in terms of their latent position. A major current concern in social networks is the emergence of polarization and filter bubbles promoting a mindset of ''us-versus-them'' that may be defined by extreme positions believed to ultimately lead to political violence and the erosion of democracy. Such polarized networks are typically characterized in terms of signed links reflecting likes and dislikes. We propose the latent Signed relational Latent dIstance Model (SLIM) utilizing for the first time the Skellam distribution as a likelihood function for signed networks and extend the modeling to the characterization of distinct extreme positions by constraining the embedding space to polytopes.


## Installation

### Create a Python 3.8.3 environment with conda

```
conda create -n ${env_name} python=3.8.3  
```

### Activate the environment

```
conda activate ${env_name} 
```

### Please install the required packages

```
pip install -r requirements.txt
```

### Additional packages

Our Pytorch implementation uses the [pytorch_sparse](https://github.com/rusty1s/pytorch_sparse) package. Installation guidelines can be found at the corresponding [Github repository](https://github.com/rusty1s/pytorch_sparse).

#### For a cpu installation please use: 

```pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.12.1+cpu.html```

#### For a gpu installation please use:

```pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.12.1+${CUDA}.html```

where ${CUDA} should be replaced by either cu102, cu113, or cu116 depending on your PyTorch installation.



## Learning embeddings for signed undirected networks using SLIM-RAA

**RUN:** &emsp; ```python main.py```

optional arguments:

**--epochs**  &emsp;  number of epochs for training (default: 5K)

**--pretrained**  &emsp;   uses pretrained embeddings for link prediction (default: True)

**--cuda**  &emsp;    CUDA training (default: True)

**--LP**   &emsp;     performs link prediction (default: True)

**--D**   &emsp;      dimensionality of the embeddings (default: 8)

**--lr**   &emsp;     learning rate for the ADAM optimizer (default: 0.05)

**--dataset** &emsp;  dataset to apply Skellam Latent Distance Modeling on (default: wiki_elec)

**--sample_percentage** &emsp;  sample size network percentage, it should be equal or less than 1 (default: 0.3)

**--reg_strength** &emsp; Regularization strength over the model parameters (default: 0.5 equivalent to normal priors) (ONLY FOR DIRECTED NETWORKS)


### Additional example for learning three-dimensional embeddings running on cpu:

**RUN:** &emsp; ```python main.py --cuda False --D 3 --pretrained False```

## For directed signed networks please use:

**RUN:** &emsp; ```python main_dir.py```



### CUDA Implementation

The code has been primarily constructed and optimized for running in a GPU-enabled environment, running the code in CPU is significantly slower.

## Reference

[Characterizing Polarization in Social Networks using the Signed Relational Latent Distance Model](https://github.com/rusty1s/pytorch_sparse](https://proceedings.mlr.press/v206/nakis23a.html)https://proceedings.mlr.press/v206/nakis23a.html), Nikolaos Nakis et al., AISTATS 23


