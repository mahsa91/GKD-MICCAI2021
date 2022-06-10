GKD: Semi-supervised Graph Knowledge Distillation for Graph-Independent Inference
====

Here is the code for node classification in graphs when the graph is not available at test time.
Ghorbani et.al. ["GKD: Semi-supervised Graph Knowledge Distillation for Graph-Independent Inference"](https://arxiv.org/pdf/2104.03597) [1]
Due to the aggregation step in graph neural networks, their performance relies on the availability of graph in both the training and testing phases. It is a common situation that the graph between nodes does not exist at inference time. GKD suggests aggregating both node features and graph using a teacher network and distilling the training knowledge to a student who only uses the node features as its input. Now the student network is able to predict the test node labels without the graph between them. Although the teacher network can be an arbitrary graph neural network, GKD suggests transferring the aggregation step from input space to label space. To this end, a modified version of the label-propagation algorithm has been provided to achieve a balance between the importance of the graph and the importance of features in the final node labels. Here is an overview of the method.
![GKD overview](https://github.com/mahsa91/GKD/blob/main/GKD.JPG?raw=true)





Usage 
------------
The main file is "main.py". Run with ```python train.py```


Input Data
------------
For running the code, you need to load data in the main.py. adjacency matrices, features, labels, training, validation, and test indices should be returned in this function. More description about each variable is as follows:
- adj: is a sparse tensor showing the **normalized** adjacency matrix between all nodes (train, validation and test). It should be noted that validation and test nodes only has self-loop without any edge to other nodes.
- Features: is a tensor that includes the features of all nodes (N by F).
- labels: is a list of labels for all nodes (with length N)
- idx_train, idx_val, idx_test: are lists of indexes for training, validation, and test samples respectively.

Parameters
------------
Here is a list of parameters that should be passed to the main function or set in the code:
- seed: seed number
- use-cuda: using CUDA for training if it is available
- epochs_teacher: number of epochs for training the teacher network (default: 300)
- epochs_student: number of epochs for training the student network (default: 200)
- epochs_lpa: number of epochs for running label-propagation algorithm (default: 10)
- lr_teacher: learning rate for the teacher network (default: 0.005)
- lr_student: learning rate for the student network (default: 0.005)
- wd_teacher: weight decay for the teacher network (default: 5e-4)
- wd_student: weight decay for the student network (default: 5e-4)
- dropout_teacher: dropout for the teacher network (default: 0.3)
- dropout_student: dropout for the student network (default: 0.3)
- burn_out_teacher: Number of epochs to drop for selecting best parameters based on validation set for teacher network (default: 100)
- burn_out_student: Number of epochs to drop for selecting best parameters based on validation set for student network (default: 100)
- alpha: a float number between 0 and 1 that shows the coefficient of remembrance term (default: 0.1)
- hidden_teacher: a list of hidden neurons in each layer of the teacher network. This variable should be set in the code (default: [8] which is a network with one hidden layer with eight neurons in it)
- hidden_student: a list of hidden neurons in each layer of the student network. This variable should be set in the code (default: [4])
- best_metric_teacher: to select the best output of teacher network, we use the performance of the network on the validation set based on this score (should be a combination between [loss, acc, f1macro] and [train, val, test]).
- best_metric_student: to select the best output of student network, we use the performance of the network on the validation set based on this score.

Metrics
------------
Accuracy, macro F1 are calculated in the code. ROAUC can be calculated for binary classification tasks.

Note
------------
Thanks to Thomas Kipf. The code is written based on the "Graph Convolutional Networks in PyTorch" [2].

Bug Report
------------
If you find a bug, please send email to mahsa.ghorbani@sharif.edu including if necessary the input file and the parameters that caused the bug.
You can also send me any comment or suggestion about the program.

References
------------
[1] [Ghorbani, Mahsa, et al. "GKD: Semi-supervised Graph Knowledge Distillation for Graph-Independent Inference." International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, Cham, 2021.](https://arxiv.org/pdf/2104.03597)

[2] [Kipf & Welling, Semi-Supervised Classification with Graph Convolutional Networks, 2016](https://arxiv.org/abs/1609.02907)

Cite
------------
Please cite our paper if you use this code in your own work:

```
@inproceedings{ghorbani2021gkd,
  title={GKD: Semi-supervised Graph Knowledge Distillation for Graph-Independent Inference},
  author={Ghorbani, Mahsa and Bahrami, Mojtaba and Kazi, Anees and Soleymani Baghshah, Mahdieh and Rabiee, Hamid R and Navab, Nassir},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={709--718},
  year={2021},
  organization={Springer}
}
```
