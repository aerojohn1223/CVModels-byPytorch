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

![Untitled](FiLM%20-%20Visual%20Reasoning%20with%20a%20General%20Conditionin%20393f975633574e3fa5dd3b43db4bf253/Untitled.png)

- i = ith input, c = cth feature.
- gamma and beta modulates neural network’s activation functions via feature-wise affine transformation.

![Untitled](FiLM%20-%20Visual%20Reasoning%20with%20a%20General%20Conditionin%20393f975633574e3fa5dd3b43db4bf253/Untitled%201.png)

- in practice, it is easier to refer to f and h as a single function that outputs one (gamma, beta) vector. (I have also done this for my implementation.)
- FiLM layers empower the FiLM generator to manipulate feature maps of a target by scaling them up or down, negating them, shutting them off, selectively thresholding them, and more.
- each feature map conditioned independently.

  

![Untitled](FiLM%20-%20Visual%20Reasoning%20with%20a%20General%20Conditionin%20393f975633574e3fa5dd3b43db4bf253/Untitled%202.png)

### Model

![Untitled](FiLM%20-%20Visual%20Reasoning%20with%20a%20General%20Conditionin%20393f975633574e3fa5dd3b43db4bf253/Untitled%203.png)

- composed of FiLM generator, FiLM-ed network, residual block, and classifier
- FiLM generator processes a question using GRU with 4096 hidden units, and 200 dimensional word embeddings.
- The visual pipeline extracts 14x14 image feature with 128 channels from a resized 224x224 input image with 3 channels. The pipeline consists of 4 layers with 128 4x4 kernels.
- Residual Block is composed of 1x1 convolution followed by ReLU, then another 3x3 convolution followed by batch normalization. After the normalization, FiLM layer is followed, with ReLU after that. The output of the first ReLU is added to the output of the second ReLU for identity mapping.
- To facilitate spatial learning, two coordinate feature maps indicating relative x and y spatial position(scaled from -1 to 1) is concatenated with each of the Residual Block’s input.
- The classifier is composed of 1x1 con layer with 512 feature maps, global max pooling, and two layer MLP with 1024 hidden units.

## What do FiLM layers learn?

![Untitled](FiLM%20-%20Visual%20Reasoning%20with%20a%20General%20Conditionin%20393f975633574e3fa5dd3b43db4bf253/Untitled%204.png)

- these images reveal that the FiLM model predicts using features of areas near answer-related or question-related objects.
- regions with question-relevant features have large activations while other regions do not, which means that appropriate feature modulation indirectly results in spatial modulation.
- In the top example, FiLM-ed network has localized the answer-referenced object alone before the MLP classifier.
- The bottom example provides evidence that the final MLP itself carries out some reasoning using FiLM to extract relevant features for its reasoning.

## Performance

![Untitled](FiLM%20-%20Visual%20Reasoning%20with%20a%20General%20Conditionin%20393f975633574e3fa5dd3b43db4bf253/Untitled%205.png)

## Implementation on Pytorch

- It was my first time implementing vision-language model on pytorch, so I looked up the codes and tried to undertand it. I fixed some of the codes that were not the same as the settings of the paper to how it is in the paper(ex. using 4 4x4 conv layers for image feature extractor).