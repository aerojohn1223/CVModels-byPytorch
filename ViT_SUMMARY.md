# An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale

Authors : Alexey Dosovitskiy et al.

Date : Jun 2021

## About Vision Transformer

- applying a standard transformer directly to images, with the fewest possible modifications
- lacks some of the inductive bias inherent to CNNs, and therefore needs larger datasets to train.
- ViT approaches beats state of the art on multiple image recognition benchmarks at the time of introduction of ViT.

## Method

<img width="759" alt="Untitled 0" src="https://github.com/aerojohn1223/CVModels-byPytorch/assets/82106824/4c7741bc-d9fc-49d9-9d9f-d4bb96c860b8">

### Image to Image Patches

<img width="478" alt="Untitled 1" src="https://github.com/aerojohn1223/CVModels-byPytorch/assets/82106824/eaeea449-64e0-4da4-b52f-7214e131e1dc">

- the image is separated to (pxp) resolution of image patches, then flattened to 2D patches.    The resulting number of patches = HW / (p*p)
- The flattened image patches are then going through linear projection, which results patch embeddings. An extra learnable classification embedding is concatenated to the very front of the sequence.
- Position Embedding is then added to the patch embeddings. The position embedding is a standard learnable 1D position embedding. Performance gain from using 2D position embedding is not observed.

<img width="665" alt="Untitled 2" src="https://github.com/aerojohn1223/CVModels-byPytorch/assets/82106824/d609f854-5e94-4ee1-ba80-905f0a78f613">

## Transformer Encoder

<img width="376" alt="Untitled 3" src="https://github.com/aerojohn1223/CVModels-byPytorch/assets/82106824/d35d771f-270b-4b37-9df8-c57cd88c0608">

- composed of layer normalization - multi head attention - layer normalization - MLP
- Layer normalization is done before multi-head attention and MLP, which is different from the original Transformer model.
    - why layer normalization? : layer normalization conducts normalization per data sample(batch normalization = normalization per feature), which means normalizing each sequence. Normalizing by every Nth place token would make less sense for sequences.
    
    ![Untitled 4](https://github.com/aerojohn1223/CVModels-byPytorch/assets/82106824/1e146141-4c31-4c7c-9365-f32ce443521b)
    
- Multi-Head attention is the same as the self attention in the original Transformer.
- MLP contains two linear layers with a GELU non-linearity.
    - what is GELU non-linearity? : stands for Gaussian Error Linear Unit. multiplies input x with the output of CDF of Gaussian distribution(input as x). By doing so, you can know how big the x is compared to the others. Also, the minimum is bounded, so it is free from gradient vanishing.
    
    ![Untitled 5](https://github.com/aerojohn1223/CVModels-byPytorch/assets/82106824/3b04a979-7715-4c41-a097-deec07f70689)
    
- the output of the transformer encoder goes through MLP head, and then classified.

## Inductive Bias

- what is inductive bias? : Inductive bias is a hypothesis that is added through information about the modality. It is used for the model to predict the correct answer when the model meets a new data.                                                                                                                                For example, CNN : patches close to each other will have similar pixels RNN : a sentence will be made from left to right, time will pass on from left to right.
- Transformer does not use inductive bias, so it is more generalized to various modalities and needs more data to train.

## Comparison with the state of the art

<img width="691" alt="Untitled 6" src="https://github.com/aerojohn1223/CVModels-byPytorch/assets/82106824/ace391b9-660d-415c-8100-b9bef925f1e5">

## Implementation on Pytorch

- I looked up how other people code to make images to image patches, but tried to use the attention codes that I used for my transformers implementation. I also tried to code with what I already know, which means I did not try to copy the codes of what other people have already made and shared. I learned a little about einops through other people’s ViT codes, and I think it would be very useful to learn einops, because the code is very intuitive.
