# **Piano Generation Using Deep Learning**

This repository contains code, notebooks and a GUI application for exploring and experimenting with various deep learning architectures to generate piano music. The experiments include recurrent architectures (GRU, LSTM), encoder-decoder models, transformer-based fine-tuning (DistilGPT-2), and GAN-based approaches.

## **Demo**

[Here](https://raw.githubusercontent.com/GecataGoranov/Piano_Generation_with_GUI/main/demo/gpt2_generated_song.mp3) is a sample of the DistilGPT-2 model

## **Table of Contents**
1. [Project Overview](#project-overview)  
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Data Representation](#data-representation)
6. [Model Architectures](#model-architectures)
7. [Results](#results)
8. [Future Work](#future-work)

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

If you wish to run the application, you would have to follow these steps:

  1. **Clone the repository and navigate to the "App/" directory**
     ```
     git clone https://github.com/GecataGoranov/Piano_Generation_with_GUI.git
     cd Piano_Generation_with_GUI/App
     ```
  3. **Install the necessary packages**
     - On **Debian/Ubuntu** systems:
       ```
       sudo apt install git-lfs ffmpeg fluidsynth
       ```
     - On **RedHat/Fedora** systems:
        ```
       sudo dnf install git-lfs ffmpeg fluidsynth
        ```
     - On **Arch** systems:
       ```
       sudo pacman -S git-lfs ffmpeg fluidsynth
       ```
     - On **Windows**:
       - Using Chocolatey (recommended)
          ```
         choco install git-lfs ffmpeg fluidsynth -y
          ```
       - Using Winget
          ```
          winget install --id=FFmpeg.FFmpeg
          winget install --id=Fluidsynth.Fluidsynth
          winget install --id=Github.GitLFS
          ```
      - On **MacOS**
        - Using Homebrew (recommended)
          ```
          brew install git-lfs ffmpeg fluidsynth
          ```
        - Using MacPorts
          ```
          sudo port -y install git-lfs ffmpeg fluidsynth
          ```
  4. **Download the large files**
     
     ```
     git lfs install
     git lfs pull
     ```
  6. **Create and activate a Python environment**
     
     ```
     python3 -m venv .venv
     source .venv/bin/activate
     ```
  8. **Install Python dependencies**
     
     ```
     pip install -r requirements.txt
     ```

## **Usage**

In order to use the application, after completing the installation process, you just have to execute the following command in the "App/" directory:
```
streamlit run main.py
```

## **Data Representation**

This project uses the MAESTRO dataset. For the various experiments, different data representation forms were used:
  - **Piano-roll representation** - The MIDI files were partitioned into timesteps, with each timestep containing a 128-dimensional vector, representing every possible pitch
  - **Word tokenization** - The MIDI files were converted into string tokens, using the `miditok` library with the REMI tokenizer
  - **Quadruplet representation** - The MIDI files were split into 4 continuous values per note: *tone length, frequency, intensity* and *time spent since the previous tone*.

## **Model Architectures**

### **GRU and LSTM**

##### **Many-to-One**

  - **Input**: Sequence of length *seq_len* with 128-dimensional piano-roll vectors
  - **Architecture**: 2 layers of GRU/LSTM with a hidden size of 256
  - **Output**: Single 128-dimensional vector predicting the next timestep

##### **Many-to-Many**
  - **Input/Output**: Seqeunces of length *seq_len*
  - **Architecture**: Same as many-to-one, but returns outputs at every timestep
  - **Note**: Did not converge; experiments discontinued

##### **Encoder-Decoder**
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

## **Results**

### **Objective Measures**

Measured against test set on:
  - **Pitch diversity**
  - **Rhythmic consistency**
  - **Note density**

The *DistilGPT-2* model achieved the highest overall scores. The Many-to-One GRU/LSTM models performed competitively on rhythmic entropy and note density, respectively.

### **Subjective Evaluation**

Listening tests revealed, that the *DistilGPT-2* model produced the most coherent samples, while the RNN-based models exhibited random pauses and noise bursts. Every user can evaluate the models for themselves by using the GUI application, containing the models.

## **Future Work**
Here are some of the ideas to work on in the future:
  - Integrate *attention mechanisms* into the encoder-decoder framework
  - Resolve and extend *GAN* training
  - Explore *larger transformer* variants

## **References**
1. Curtis Hawthorne, Andriy Stasyuk, Adam Roberts, Ian Simon, Cheng-Zhi Anna Huang,
  Sander Dieleman, Erich Elsen, Jesse Engel, and Douglas Eck. *"Enabling
  Factorized Piano Music Modeling and Generation with the MAESTRO Dataset."*
  In International Conference on Learning Representations, 2019.

2. Olof Morgen. *"C-RNN-GAN: Continuous recurrent neural networks with adversarial training"*

3. Yu-Siang Huang, Yi-Hsuan Yang. *"Pop Music Transformer: Beat-based Modeling and Generation of Expressive Pop Piano Compositions"*

4. Sanh, Victor and Debut, Lysandre and Chaumond, Julien and Wolf, Thomas. *"DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter"*
