import pandas as pd
import os
import muspy
import random
import torch
import torch.nn as nn
import streamlit as st

from tqdm import tqdm
from torch.utils.data import Dataset


class MusicDatasetManyToOne(Dataset):
    def __init__(self, metadata, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.songs = []

        for file in metadata["midi_filename"]:
            file_path = os.path.join("../MAESTRO dataset/maestro-v3.0.0-midi/maestro-v3.0.0/", file)
            muspy_file = muspy.read(file_path)
            self.songs.append(muspy_file)

    def __len__(self):
        return len(self.songs)

    def __getitem__(self, idx):
        pianoroll_song = self.songs[idx].to_pianoroll_representation()
        pianoroll_song = torch.tensor(pianoroll_song, dtype=torch.float32)

        start_idx = random.randint(0, len(pianoroll_song) - self.seq_len - 1)

        input_seq = pianoroll_song[start_idx:start_idx+self.seq_len]

        target = pianoroll_song[start_idx+self.seq_len]

        return input_seq, target
    


class ManyToOneGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(ManyToOneGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.gru(x, h0)
        out = out[:, -1, :]
        out = self.fc(out)

        return out
    

class ManyToOneLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(ManyToOneLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, (h_n, c_n) = self.lstm(x, (h0, c0))
        out = self.fc(h_n[-1])

        return out
    

def generate_music_many_to_one(model, initial_sequence, seq_len=100, max_generate_len=5000):
    bar = st.progress(0)
    input_seq = initial_sequence
    generated_music = input_seq.squeeze(0)
    input_seq = input_seq.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(max_generate_len)):
            output = model(input_seq)
            next_step = output
            generated_music = torch.cat((generated_music, next_step), dim=0)
            input_seq = generated_music[-seq_len:].unsqueeze(0)
            bar.progress(i * 100 // max_generate_len)

    return generated_music
