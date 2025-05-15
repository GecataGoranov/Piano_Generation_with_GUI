import streamlit as st
import pandas as pd
import torch
import random

from pathlib import Path
from tqdm import tqdm
from utils.conversions import midi_to_wav, pianoroll_to_midi
from utils.many_to_one import MusicDatasetManyToOne, ManyToOneGRU, ManyToOneLSTM, generate_music_many_to_one
from utils.encoder_decoder import MusicDataset, EncoderLSTM, DecoderLSTM, Seq2SeqLSTM, generate_music_many_to_many
from transformers import GPT2LMHeadModel
from miditok import REMI, TokenizerConfig, TokSequence


@st.cache_data
def get_metadata(split):
    all_data = pd.read_csv("MAESTRO dataset/maestro-v3.0.0.csv")
    return all_data[all_data["split"] == split]


@st.cache_data
def load_dataset(type):
    dataset = torch.load(f"datasets/music_dataset_test_{type}_muspy.pt", weights_only=False)
    return dataset

@st.cache_data
def load_many_to_one_gru():
    model = ManyToOneGRU(128, 256, 128, 2)
    model.load_state_dict(torch.load("models/many_to_one_gru_epoch_150.pt", map_location=torch.device('cpu')))
    return model


@st.cache_data
def load_many_to_one_lstm():
    model = ManyToOneLSTM(128, 256, 128, 2)
    model.load_state_dict(torch.load("models/many_to_one_lstm_epoch_150.pt", map_location=torch.device('cpu')))
    return model

@st.cache_data
def load_encoder_decoder_lstm():
    encoder = EncoderLSTM(input_size=128, hidden_size=1024, num_layers=2)
    decoder = DecoderLSTM(output_size=128, hidden_size=1024, num_layers=2)

    model = Seq2SeqLSTM(encoder, decoder)
    model.load_state_dict(torch.load("models/encoder_decoder_lstm_epoch_120.pt", map_location=torch.device("cpu")))
    return model

@st.cache_data
def get_tokenizer():
    tokenizer_config = TokenizerConfig(
    num_velocities=16,
    use_chords=True,
    use_programs=False,
    use_rests=True,
    use_tempos=True,
    use_time_signatures=False,
    use_sustain_pedals=True,
    )

    tokenizer = REMI(tokenizer_config)
    return tokenizer

@st.cache_data
def load_gpt2():
    model = GPT2LMHeadModel.from_pretrained("models/pretrained_gpt2_epoch_11")
    tokenizer = get_tokenizer()
    model.resize_token_embeddings(len(tokenizer.vocab))

    return model

def generate_music_gpt2(model, tokenizer):
    model.eval()

    input_ids = torch.tensor([[tokenizer.vocab['BOS_None']]])

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_length=512,
            temperature=1.0,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.vocab['PAD_None'],
            eos_token_id=tokenizer.vocab['EOS_None'],
            attention_mask=torch.ones_like(input_ids)
        )
    
    return output[0].tolist()

def get_music_many_to_one(model):
    dataset = load_dataset("many_to_one")
    initial_seq = dataset[random.randint(0, len(dataset)-1)][0]
    generated_music = generate_music_many_to_one(model, initial_seq, seq_len=100, max_generate_len=1000)
    midi = pianoroll_to_midi(generated_music, threshold=0.1)
    wav = midi_to_wav(midi)
    return wav

def get_music_encoder_decoder(model):
    dataset = load_dataset("many_to_many")
    initial_seq = dataset[random.randint(0, len(dataset)-1)][0]
    generated_music = generate_music_many_to_many(model, initial_seq, seq_len=100, max_generate_len=500)
    midi = pianoroll_to_midi(generated_music, time_step=0.2, threshold=0.1)
    wav = midi_to_wav(midi)
    return wav

def get_music_gpt2(generated_ids, tokenizer):
    reverse_vocab = {v: k for k, v in tokenizer.vocab.items()}
    generated_tokens = [reverse_vocab.get(token_id, "[UNK]") for token_id in generated_ids]
    tok_seq = TokSequence(tokens=generated_tokens, ids=generated_ids)
    tokenizer.complete_sequence(tok_seq)
    midi = tokenizer([tok_seq])
    midi.dump_midi("temp/temp_midi.mid")
    wav = midi_to_wav("temp/temp_midi.mid", volume=15.0)
    return wav

def show_download_button():
    st.text("Like what you heard? Download it from here:")
    file_path = Path("temp/temp_wav.wav")

    with file_path.open("rb") as f:
        st.download_button(
            label="Download WAV",
            data=f,
            file_name="tone.wav",
            mime="audio/wav"
        )


def main():
    st.set_page_config(
        page_title="Music Generation Demo",
        page_icon=":musical_note:",
        layout="centered"
    )

    st.markdown(
        """
        <div style="text-align: center; margin-top: 20px;">
            <h1>Try out my models! 🎶</h1>
            <p style="font-size: 1.1rem;">
                Explore different neural network architectures for music generation.
                Just click a button below to hear what each model comes up with.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Many-to-One Models")

        if st.button("Many-to-One GRU"):
            with st.spinner("Generating..."):
                model = load_many_to_one_gru()
                get_music_many_to_one(model)
            st.success("Done!")
            st.audio("temp/temp_wav.wav")
            show_download_button()


        if st.button("Many-to-One LSTM"):
            with st.spinner("Generating..."):
                model = load_many_to_one_lstm()
                get_music_many_to_one(model)
            st.success("Done!")
            st.audio("temp/temp_wav.wav")
            show_download_button()

    with col2:
        st.subheader("Sequence-to-Sequence & GPT-2")

        if st.button("Encoder-Decoder LSTM"):
            with st.spinner("Generating..."):
                model = load_encoder_decoder_lstm()
                get_music_encoder_decoder(model)
            st.success("Done!")
            st.audio("temp/temp_wav.wav")
            show_download_button()

        if st.button("DistilGPT-2"):
            with st.spinner("Generating..."):
                model = load_gpt2()
                tokenizer = get_tokenizer()
                generated_ids = generate_music_gpt2(model, tokenizer)
                get_music_gpt2(generated_ids, tokenizer)
            st.success("Done!")
            st.audio("temp/temp_wav.wav")
            show_download_button()


if __name__ == "__main__":
    main() 