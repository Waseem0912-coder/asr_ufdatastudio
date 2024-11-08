import os
import pandas as pd
import torchaudio
import torch
from g2p_en import G2p
import numpy as np
from scipy.stats import norm

def convert_to_phonemes(text, g2p):
    try:
        phonemes = g2p(text)
        return " ".join(phonemes)
    except Exception as e:
        return None

def phonetic_alignment(audio_filepath, phonetic_transcript, model):
    try:
        waveform, sample_rate = torchaudio.load(audio_filepath)
        waveform = waveform.to('cuda')

        with torch.inference_mode():
            features_list, _ = model.extract_features(waveform)
        features = features_list[0].cpu().numpy()

        phonemes = phonetic_transcript.split()
        num_frames = features.shape[1]
        num_states = len(phonemes)
        frame_duration = waveform.shape[1] / sample_rate / num_frames
        
        emissions = np.zeros((num_states, num_frames))
        for i, phoneme in enumerate(phonemes):
            emissions[i, :] = norm.pdf(np.arange(num_frames), loc=num_frames * (i + 1) / (num_states + 1), scale=num_frames / (2 * num_states))

        viterbi = np.zeros((num_states, num_frames))
        backpointer = np.zeros((num_states, num_frames), dtype=int)
        viterbi[:, 0] = emissions[:, 0]

        for t in range(1, num_frames):
            for s in range(num_states):
                max_tr_prob = viterbi[:, t - 1] * emissions[s, t]
                backpointer[s, t] = np.argmax(max_tr_prob)
                viterbi[s, t] = max_tr_prob[backpointer[s, t]]

        best_path = np.zeros(num_frames, dtype=int)
        best_path[-1] = np.argmax(viterbi[:, -1])
        for t in range(num_frames - 2, -1, -1):
            best_path[t] = backpointer[best_path[t + 1], t + 1]

        aligned_phonemes = []
        current_phoneme = phonemes[best_path[0]]
        start_time = 0.0

        for t in range(1, num_frames):
            phoneme = phonemes[best_path[t]]
            if phoneme != current_phoneme:
                end_time = t * frame_duration
                aligned_phonemes.append({"phoneme": current_phoneme, "start_time": start_time, "end_time": end_time})
                current_phoneme = phoneme
                start_time = end_time

        end_time = num_frames * frame_duration
        aligned_phonemes.append({"phoneme": current_phoneme, "start_time": start_time, "end_time": end_time})

        alignment_result = {
            "audio_filepath": audio_filepath,
            "phonetic_transcript": phonetic_transcript,
            "alignment": aligned_phonemes
        }
        return alignment_result
    except Exception as e:
        return None

if __name__ == "__main__":
    data = pd.read_csv("/blue/ufdatastudios/ahmed.waseem/ctc/meta_speaker.csv")
    data = data[data['duration'].apply(lambda x: x >= 5)]

    audio_folder = "/blue/ufdatastudios/ahmed.waseem/processed_audio"
    data["audio_filepath"] = data["audio_filepath"].apply(lambda x: os.path.join(audio_folder, x))

    g2p = G2p()
    data['phonetic_transcript'] = data['text'].apply(lambda x: convert_to_phonemes(x, g2p))
    data = data.dropna(subset=['phonetic_transcript'])

    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().to('cuda')

    alignment_results = []
    total_files = len(data)

    for index, row in data.iterrows():
        result = phonetic_alignment(row['audio_filepath'], row['phonetic_transcript'], model)
        if result:
            alignment_results.append(result)

    alignment_df = pd.DataFrame(alignment_results)
    alignment_df.to_csv("phonetic_alignment_results.csv", index=False)

