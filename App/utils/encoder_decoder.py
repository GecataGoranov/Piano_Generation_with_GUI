import torch
import torch.nn as nn
import random
import os
import muspy

from torch.utils.data import Dataset


class MusicDataset(Dataset):
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

        start_idx = random.randint(0, len(pianoroll_song) - 2 * self.seq_len)

        input_seq = pianoroll_song[start_idx:start_idx+self.seq_len]

        target = pianoroll_song[start_idx+(self.seq_len // 2) : start_idx+self.seq_len+(self.seq_len // 2)]

        return input_seq, target
    

class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        _, (h_n, c_n) = self.lstm(x)

        h_n = h_n.view(self.num_layers, 2, x.size(0), self.hidden_size)
        c_n = c_n.view(self.num_layers, 2, x.size(0), self.hidden_size)

        h_n = h_n.sum(dim=1)
        c_n = c_n.sum(dim=1)

        return h_n, c_n
    

class DecoderLSTM(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(output_size, hidden_size, num_layers, batch_first=True)
        self.ff = nn.Linear(hidden_size, output_size)

    def forward(self, x, h_n, c_n):
        output, (h_n, c_n) = self.lstm(x, (h_n, c_n))
        output = self.ff(output)
        return output, h_n, c_n
    

class Seq2SeqLSTM(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2SeqLSTM, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, target_seq_len, teacher_forcing_ratio=0.5):

        batch_size = x.size(0)
        output_size = self.decoder.ff.out_features
        outputs = torch.zeros(batch_size, target_seq_len, output_size).to(x.device)

        h_n, c_n = self.encoder(x)

        decoder_input = torch.zeros(batch_size, 1, output_size).to(x.device)

        for t in range(target_seq_len):
            output, h_n, c_n = self.decoder(decoder_input, h_n, c_n)
            outputs[:, t, :] = output.squeeze(1)

            if torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = x[:, t, :].unsqueeze(1)
            else:
                decoder_input = output

        return outputs
    

def generate_music_many_to_many(model, initial_sequence, seq_len=100, max_generate_len=5000):
    input_seq = initial_sequence
    input_seq = input_seq.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(input_seq, max_generate_len, 0.0)
        return output.squeeze()