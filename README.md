# Stochastic Adversarial Video Prediction
[[Project Page]](https://alexlee-gk.github.io/video_prediction/) [[Paper]](https://arxiv.org/abs/1804.01523)

TensorFlow implementation for stochastic adversarial video prediction. Given a sequence of initial frames, our model is able to predict future frames of various possible futures.

**Stochastic Adversarial Video Prediction,**  
[Alex X. Lee](https://people.eecs.berkeley.edu/~alexlee_gk/), [Richard Zhang](https://richzhang.github.io/), [Frederik Ebert](https://febert.github.io/), [Pieter Abbeel](https://people.eecs.berkeley.edu/~pabbeel/), [Chelsea Finn](https://people.eecs.berkeley.edu/~cbfinn/), [Sergey Levine](https://people.eecs.berkeley.edu/~svlevine/).  
arXiv preprint arXiv:1804.01523, 2018.

## Prerequisites
- Linux or macOS
- Python 2 or 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started ###
### Installation
- Clone this repo:
```bash
git clone -b master --single-branch https://github.com/alexlee-gk/video_prediction.git
cd video_prediction
```
- Install TensorFlow >= 1.5 and dependencies from http://tensorflow.org/
- Install ffmpeg (optional, used to generate GIFs for visualization, e.g. in TensorBoard)
- Install other dependencies
```bash
pip install -r requirements.txt
```

### Use a Pre-trained Model
- Download and preprocess a dataset (e.g. `bair`):
```bash
bash ./data/download_and_preprocess_dataset.sh bair
```
- Download a pre-trained model (e.g. `savp`) for that dataset:
```bash
bash ./models/download_model.sh bair savp
```

### Model Training


### Datasets
Download the datasets using the following script. These datasets are collected by other researchers. Please cite their papers if you use the data.
- Download and preprocess the dataset.
```bash
bash ./data/download_and_preprocess_dataset.sh dataset_name
```
- `bair`: [BAIR robot pushing dataset](https://sites.google.com/view/sna-visual-mpc/). [[Citation](data/bibtex/sna.txt)]
- `kth`: [KTH human actions dataset](http://www.nada.kth.se/cvap/actions/). [[Citation](data/bibtex/kth.txt)]

## Models


### Citation

If you find this useful for your research, please use the following.

```
@article{lee2018savp,
  title={Stochastic Adversarial Video Prediction},
  author={Alex X. Lee and Richard Zhang and Frederik Ebert and Pieter Abbeel and Chelsea Finn and Sergey Levine},
  journal={arXiv preprint arXiv:1804.01523},
  year={2018}
}
```
