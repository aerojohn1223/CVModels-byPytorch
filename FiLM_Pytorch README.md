# FiLM - Visual Reasoning with a General Conditioning

Authors : Ethan Perez, Florian Strub, Harm de Vries, Vincent Dumoulin, Aaron Courville

Date of Publish : 18 Dec 2017

Dataset : CLEVR Dataset

## About FiLM

- helps to achieve strong visual reasoning.
- FiLM layers influence neural network computation via “feature-wise affine transformation” based on conditioning information.
- enables a RNN over an input question to influence CNN computation over an image.

## Method

### Feature-wise Linear Modulation

- FiLM learns functions f and h which output gamma and beta as a function of input x.

<img width="837" alt="Untitled" src="https://github.com/aerojohn1223/CVModels-byPytorch/assets/82106824/8cf15486-e479-4632-a254-bda99a7e51b1">

- i = ith input, c = cth feature.
- gamma and beta modulates neural network’s activation functions via feature-wise affine transformation.

<img width="809" alt="Untitled 1" src="https://github.com/aerojohn1223/CVModels-byPytorch/assets/82106824/82ee303f-42df-4379-ba04-81d6e903359d">


- in practice, it is easier to refer to f and h as a single function that outputs one (gamma, beta) vector. (I have also done this for my implementation.)
- FiLM layers empower the FiLM generator to manipulate feature maps of a target by scaling them up or down, negating them, shutting them off, selectively thresholding them, and more.
- each feature map conditioned independently.

  

<img width="794" alt="Untitled 2" src="https://github.com/aerojohn1223/CVModels-byPytorch/assets/82106824/d2fef94d-a868-4252-ba80-17cac171b285">


### Model

<img width="577" alt="Untitled 3" src="https://github.com/aerojohn1223/CVModels-byPytorch/assets/82106824/90db72e4-70d9-4805-9809-be60ae8b6915">


- composed of FiLM generator, FiLM-ed network, residual block, and classifier
- FiLM generator processes a question using GRU with 4096 hidden units, and 200 dimensional word embeddings.
- The visual pipeline extracts 14x14 image feature with 128 channels from a resized 224x224 input image with 3 channels. The pipeline consists of 4 layers with 128 4x4 kernels.
- Residual Block is composed of 1x1 convolution followed by ReLU, then another 3x3 convolution followed by batch normalization. After the normalization, FiLM layer is followed, with ReLU after that. The output of the first ReLU is added to the output of the second ReLU for identity mapping.
- To facilitate spatial learning, two coordinate feature maps indicating relative x and y spatial position(scaled from -1 to 1) is concatenated with each of the Residual Block’s input.
- The classifier is composed of 1x1 con layer with 512 feature maps, global max pooling, and two layer MLP with 1024 hidden units.

## What do FiLM layers learn?

<img width="658" alt="Untitled 4" src="https://github.com/aerojohn1223/CVModels-byPytorch/assets/82106824/0b08e3c0-b92b-442f-a0d1-0a81c976cf1a">


- these images reveal that the FiLM model predicts using features of areas near answer-related or question-related objects.
- regions with question-relevant features have large activations while other regions do not, which means that appropriate feature modulation indirectly results in spatial modulation.
- In the top example, FiLM-ed network has localized the answer-referenced object alone before the MLP classifier.
- The bottom example provides evidence that the final MLP itself carries out some reasoning using FiLM to extract relevant features for its reasoning.

## Performance

<img width="1075" alt="Untitled 5" src="https://github.com/aerojohn1223/CVModels-byPytorch/assets/82106824/43ce7c10-5541-4e73-8965-38a4c5a4acc2">

## Implementation on Pytorch

- It was my first time implementing vision-language model on pytorch, so I looked up the codes and tried to undertand it. I fixed some of the codes that were not the same as the settings of the paper to how it is in the paper(ex. using 4 4x4 conv layers for image feature extractor).
