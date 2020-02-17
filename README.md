# Scaling Robot Supervision to Hundreds of Hours with RoboTurk: Robotic Manipulation Dataset through Human Reasoning and Dexterity

Created by [PAIR Group](http://pair.stanford.edu/) from Stanford University


## Introduction

This library provides scripts to parse the HDF5 file and aligned with the videos in the video directory. We also provide scripts to process and split the dataset for video prediction (SV2P: [https://arxiv.org/abs/1710.11252]) along with the necessary changes needed in tensor2tensor to match the hyperparameters in the original paper.

## Citation

If you find our work useful in your research, please consider citing:

```
@misc{m2019scaling,
    title={Scaling Robot Supervision to Hundreds of Hours with RoboTurk: Robotic Manipulation Dataset through Human Reasoning and Dexterity},
    author={Ajay Mandlekar and Jonathan Booher and Max Spero and Albert Tung and Anchit Gupta and Yuke Zhu and Animesh Garg and Silvio Savarese and Li Fei-Fei},
    year={2019},
    eprint={1911.04052},
    archivePrefix={arXiv},
    primaryClass={cs.RO}
}
```

## Installation

[Tensorflow](https://github.com/tensorflow/tensorflow) and [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor). It is required that you have access to GPUs to train. We will also provide a virtual environment installation package.

The code is tested with Python 3.6.

## Parsing the Dataset and Training Video Prediction

We have outlined a more detailed wiki page with associated code blocks here that can help you parse the dataset and documents the video prediction pipeline: https://github.com/StanfordVL/roboturk_real_dataset/wiki

## License

This dataset and codebase are released under the MIT License.

## Change Log

