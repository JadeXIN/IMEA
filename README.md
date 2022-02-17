# [Informed Multi-context Entity Alignment (2022 WSDM)](https://arxiv.org/pdf/2201.00304)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg?style=flat-square)](https://github.com/JadeXIN/IMEA/issues)
[![language-python3](https://img.shields.io/badge/Language-Python3-blue.svg?style=flat-square)](https://www.python.org/)
[![made-with-Tensorflow](https://img.shields.io/badge/Made%20with-Tensorflow-orange.svg?style=flat-square)](https://www.tensorflow.org/)

> Entity alignment is a crucial step in integrating knowledge graphs (KGs) from multiple sources. 
Previous attempts at entity alignment have explored different KG structures, such as neighborhood-based and path-based contexts, to learn entity embeddings, but they are limited in capturing the multi-context features. Moreover, most approaches directly utilize the embedding similarity to determine entity alignment without considering the global interaction among entities and relations. In this work, we propose an Informed Multi-context Entity Alignment (IMEA) model to address these issues. In particular, we introduce Transformer to flexibly capture the relation, path, and neighborhood contexts, and design holistic reasoning to estimate alignment probabilities based on both embedding similarity and the relation/entity functionality. The alignment evidence obtained from holistic reasoning is further injected back into the Transformer via the proposed soft label editing to inform embedding learning. Experimental results on several benchmark datasets demonstrate the superiority of our IMEA model compared with existing state-of-the-art entity alignment methods. 


## Overview

We build our model based on [Python](https://www.python.org/) and [Tensorflow](https://www.tensorflow.org/). Our implementation follows [OpenEA](https://github.com/nju-websoft/OpenEA).

### Getting Started
Before starting our implementation, please follow [OpenEA](https://github.com/nju-websoft/OpenEA) to complete the installation of OpenEA Library, and we also recommend creating a new conda enviroment for our model.

#### Package Description

```
src/
├── openea/
│   ├── approaches/: package of the implementations for existing embedding-based entity alignment approaches
│   ├── models/: package of the implementations for unexplored relationship embedding models
│   ├── modules/: package of the implementations for the framework of embedding module, alignment module, and their interaction
│   ├── expriment/: package of the implementations for evalution methods
```

#### Dependencies
* Python 3.x (tested on Python 3.6)
* Tensorflow 1.x (tested on Tensorflow 1.8)
* Scipy
* Numpy
* Graph-tool or igraph or NetworkX
* Pandas
* Scikit-learn
* Matching==0.1.1
* Gensim


#### Usage
The following is an example about how to run IMEA in Python (We assume that you have already downloaded our [datasets](https://www.dropbox.com/s/hbyzesmz1u7ejdu/OpenEA_dataset.zip?dl=0) and configured the hyperparameters as in the our config file.).)

To run the off-the-shelf approaches on our datasets and reproduce our experiments, change into the ./run/ directory and use the following script:

```bash
python main_from_args.py "predefined_arguments" "dataset_name" "split"
```

For example, if you want to run IMEA on D-W-15K (V1) using the first split, please execute the following script:

```bash
python main_from_args.py ./args/transformer4ea_args_15K.json D_W_15K_V1 721_5fold/1/
```

### Dataset

We use the benchmark dataset released on OpenEA.

*#* Entities | Languages | Dataset names
:---: | :---: | :---: 
15K | Cross-lingual | EN-FR-15K, EN-DE-15K
15K | English | D-W-15K, D-Y-15K
100K | Cross-lingual | EN-FR-100K, EN-DE-100K
100K | English-lingual | D-W-100K, D-Y-100K

The datasets can be downloaded from [here](https://www.dropbox.com/s/hbyzesmz1u7ejdu/OpenEA_dataset.zip?dl=0).


## Citation
If you find the implementation of our model or the experimental results useful, please kindly cite the following paper:
```
@article{xin2022informed,
  title={Informed Multi-context Entity Alignment},
  author={Xin, Kexuan and Sun, Zequn and Hua, Wen and Hu, Wei and Zhou, Xiaofang},
  journal={arXiv preprint arXiv:2201.00304},
  year={2022}
}

```

## Acknowledgement
We refer to the codes of these repos: [OpenEA](https://github.com/nju-websoft/OpenEA). 
Thanks for their great contributions!
