import subprocess
import pandas as pd 
import os


df = pd.read_csv("/blue/ufdatastudios/ahmed.waseem/ctc/meta_speaker.csv")
data = df[df['duration'].apply(lambda x: x >= 5)]
data = data[data['text'].apply(lambda x: len(x.split()) > 3)]
audio_folder = "/blue/ufdatastudios/ahmed.waseem/processed_audio"
data["audio_filepath"] = data["audio_filepath"].apply(lambda x: os.path.join(audio_folder, x))

def save_text_to_file(df, file_path):
    with open(file_path, "w", encoding="utf-8") as file:
        for line in df["text"]:
            file.write(line + "\n")

def run_mfa_g2p(input_file, output_file,num_jobs=4):
    try:
        subprocess.run(
            ["mfa", "g2p", "--num_jobs", str(num_jobs), input_file, "english_india_mfa", output_file],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error running MFA G2P: {e}")

def load_phonemes_from_file(file_path):
    phonemes = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            phonemes.append(line.strip())
    return phonemes

input_file_path = "input_text.txt"
output_file_path = "output_phonemes.txt"

save_text_to_file(data, input_file_path)

run_mfa_g2p(input_file_path, output_file_path)

data["phonemes"] = load_phonemes_from_file(output_file_path)

data.to_pickle("g2p.pkl")