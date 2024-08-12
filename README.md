# Model-Agnostic Meta-Learning (MAML)

This repository contains the TensorFlow implementation of the MAML algorithm, as described in the paper "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks" by Finn, Abbeel, and Levine. MAML is designed for rapid adaptation to new tasks with a small amount of data.

## Introduction

MAML aims to train models that can quickly adapt to new tasks with minimal training on the new task's data. This implementation specifically focuses on the application of MAML to few-shot learning tasks using datasets like Omniglot and MiniImagenet.

## Prerequisites

- Python 3.7
- TensorFlow 1.15.0
- Numpy
- Pandas


## Installation

First, clone the repository to your local machine:


```bash
git clone https://github.com/yourusername/maml-tensorflow.git
cd maml-tensorflow
```


Then, install the required dependencies:


```bash
pip install -r requirements.txt
```


##Usage

To train the MAML model on the Omniglot dataset, you can use the following command:


```bash
python main.py --datasource=omniglot --num_classes=5 --update_batch_size=1 --meta_lr=0.001 --update_lr=0.01 --num_updates=5 --metatrain_iterations=15000 --meta_batch_size=25
python main.py --datasource=miniimagenet --num_classes=5 --update_batch_size=1 --meta_lr=0.001 --update_lr=0.01 --num_updates=5 --metatrain_iterations=15000 --meta_batch_size=25
```

## Configuration
You can adjust the hyperparameters and settings of the model by modifying the flags in the main.py file or by specifying them on the command line as shown above.


|  Configuration |
| :------------ |
| --num_classes: Number of classes for each task. |
|  --update_batch_size: Number of examples per class per task for the inner update. |
| --meta_lr: Learning rate for meta updates.  |
|  --update_lr: Learning rate for task-specific updates. |
| --num_updates: Number of inner gradient updates during training.  |
| --metatrain_iterations: Number of iterations/tasks to train on.  |
| --meta_batch_size: Number of tasks sampled per meta-update.  |
|--datasource: The dataset to use (omniglot or miniimagenet).|


##Extending
This framework can be extended to other datasets or adapted for different meta-learning tasks. You might need to implement custom data loaders and modify the DataGenerator class accordingly.

## Citation
If you use this code for your research, please cite the original paper[ Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks (Finn et al., ICML 2017)](https://arxiv.org/abs/1703.03400 " Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks (Finn et al., ICML 2017)"):
```css
@article{finn2017model,
  title={Model-agnostic meta-learning for fast adaptation of deep networks},
  author={Finn, Chelsea and Abbeel, Pieter and Levine, Sergey},
  journal={arXiv preprint arXiv:1703.03400},
  year={2017}
}

```