# **Piano Generation Using Deep Learning**

This repository contains code, notebooks and a GUI application for exploring and experimenting with various deep learning architectures to generate piano music. The experiments include recurrent architectures (GRU, LSTM), encoder-decoder models, transformer-based fine-tuning (DistilGPT-2), and GAN-based approaches.

# **Table of Contents**

## TODO

## **Project Overview**

This project investigates several families of deep learning models:
  - **Recurrent Neural Networks (GRU, LSTM)** in many-to-one, many-to-many and encoder-decoder setups using piano-roll data representation
  - **Transformer-based** fine-tuning of *DistilGPT-2* with musical tokenization (REMI)
  - **Generative Adversarial Networks (GAN)**, combining LSTM-based generator and discriminator
Each approach is implemented in a separate Jupyter notebook, showing all of the experiments, conducted step-by-step.

## **Features**

  - Preprocessing routines for piano-roll data representation
  - Implementation of multiple RNN-based architectures
  - Encoder-Decoder model with bidirectional LSTM encoder and teacher forcing in decoder
  - Fine-tuning *DistilGPT-2* on musical tokens
  - GAN experiments, following C-RNN-GAN design
  - Objective and subjective evaluation scripts for pitch diversity, rhythmic consistency, note density and audio quality
  - Interactive GUI to generate and listen to piano sequences

## **Installation**

### TODO

## **Usage**

### TODO

## **Model Architectures**

### **GRU and LSTM**

#### **Many-to-One**

  - **Input**: Sequence of length *seq_len* with 128-dimensional piano-roll vectors
  - **Architecture**: 2 layers of GRU/LSTM with a hidden size of 256
  - **Output**: Single 128-dimensional vector predicting the next timestep

#### **Many-to-Many**
  - **Input/Output**: Seqeunces of length *seq_len*
  - **Architecture**: Same as many-to-one, but returns outputs at every timestep
  - **Note**: Did not converge; experiments discontinued

#### **Encoder-Decoder**
  - **Encoder**: 2 layers of bidirectional LSTM with a hidden size of 1024, input dimension - 128
  - **Decoder**: LSTM + linear projection to 128 dimensions with teacher forcing
  - **Workflow**: Encode full sequence, then decode iteratively to generate music

### **Pre-trained GPT-2 Fine Tuning**
  - **Model**: *DistilGPT-2* from HuggingFace Transformers
  - **Tokenizer**: REMI musical tokens
  - **Outcome**: Best performance on objective/subjective metrics

### **GAN Architecture**
  - **Generator**: 100-dimensional input -> 2 LSTM layers with a hidden size of 350 -> linear layer to 4-dimensional representation
  - **Discriminator**: 2 bidirectional LSTM layers with a hidden size of 350 -> linear output
  - **Status**: Training script implemented, but unresolved errors prevented training
