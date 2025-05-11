import pretty_midi
import wave
import io
import subprocess

from midi2audio import FluidSynth


def pianoroll_to_midi(pianoroll, time_step=0.1, threshold=0.01):
    pianoroll = pianoroll.numpy()
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    current_pitches = {}

    for i in range(pianoroll.shape[1]):
        current_pitches[i] = 0

    for time in range(pianoroll.shape[0]):
        print(time)
        for pitch in range(pianoroll.shape[1]):
            if pianoroll[time, pitch] > threshold and current_pitches[pitch] == 0:
                current_pitches[pitch] = time
            if current_pitches[pitch] != 0 and pianoroll[time, pitch] <= threshold:
                note = pretty_midi.Note(velocity=100, pitch=pitch, start=current_pitches[pitch] * time_step, end=(time + 1) * time_step)
                instrument.notes.append(note)
                current_pitches[pitch] = 0

    midi.instruments.append(instrument)
    midi.write("temp/temp_midi.mid")
    return "temp/temp_midi.mid"


def increase_volume(input_wav, output_wav, volume=2.0):
    subprocess.run([
        "ffmpeg", "-y",
        "-i", input_wav,
        "-filter:a", f"volume={volume}",
        output_wav
    ], check=True)


def midi_to_wav(midi, volume=2.0):
    fs = FluidSynth("soundfonts/FluidR3_GM.sf2")
    fs.midi_to_audio(midi, "temp/temp_wav_quiet.wav")
    increase_volume("temp/temp_wav_quiet.wav", "temp/temp_wav.wav", volume=volume)