import torch
import torchaudio
import pandas as pd
import IPython
import matplotlib.pyplot as plt
import torchaudio.functional as F
import os

data = df = pd.read_csv("/blue/ufdatastudios/ahmed.waseem/ctc/meta_speaker.csv")
data = data[data['duration'].apply(lambda x: x >= 5)]
audio_folder = "/blue/ufdatastudios/ahmed.waseem/processed_audio"
data["audio_filepath"] = data["audio_filepath"].apply(lambda x: os.path.join(audio_folder, x))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

bundle = torchaudio.pipelines.MMS_FA
model = bundle.get_model(with_star=False).to(device)
LABELS = bundle.get_labels(star=None)

dictionary_path = "/home/ahmed.waseem/Documents/MFA/pretrained_models/dictionary/english_india_mfa.dict"
DICTIONARY = {}
with open(dictionary_path, "r") as f:
    for line in f:
        word, *phonemes = line.strip().split()
        if word not in DICTIONARY:
            DICTIONARY[word] = len(DICTIONARY)

DICTIONARY["*"] = len(DICTIONARY)

def adjust_emission(emission, dictionary_size):
    if emission.size(-1) < dictionary_size:
        pad_size = dictionary_size - emission.size(-1)
        padding = torch.zeros((emission.size(0), emission.size(1), pad_size), device=emission.device)
        emission = torch.cat((emission, padding), dim=-1)
    return emission

def align(emission, tokens):
    targets = torch.tensor([tokens], dtype=torch.int32, device=device)
    alignments, scores = F.forced_align(emission, targets, blank=0)
    alignments, scores = alignments[0], scores[0]
    scores = scores.exp()
    return alignments, scores

def compute_alignments(emission, transcript, dictionary):
    tokens = [dictionary[char] for word in transcript for char in word if char in dictionary]
    alignment, scores = align(emission, tokens)
    token_spans = F.merge_tokens(alignment, scores)
    if len(token_spans) != sum(len(word) for word in transcript if all(char in dictionary for char in word)):
        raise ValueError("Mismatch between token spans and transcript length. Please check the dictionary and transcript.")
    word_spans = unflatten(token_spans, [len(word) for word in transcript if all(char in dictionary for char in word)])
    return word_spans

def unflatten(list_, lengths):
    assert len(list_) == sum(lengths), f"Mismatch: len(list_)={len(list_)} and sum(lengths)={sum(lengths)}"
    i = 0
    ret = []
    for l in lengths:
        ret.append(list_[i : i + l])
        i += l
    return ret

def _score(spans):
    return sum(span.score for span in spans) / len(spans)

def plot_alignments(waveform, token_spans, emission, transcript, sample_rate=bundle.sample_rate):
    ratio = waveform.size(1) / emission.size(1) / sample_rate

    fig, axes = plt.subplots(2, 1)
    axes[0].imshow(emission[0].detach().cpu().T, aspect="auto")
    axes[0].set_title("Emission")
    axes[0].set_xticks([])

    axes[1].specgram(waveform[0], Fs=sample_rate)
    for t_spans, chars in zip(token_spans, transcript):
        t0, t1 = t_spans[0].start + 0.1, t_spans[-1].end - 0.1
        axes[0].axvspan(t0 - 0.5, t1 - 0.5, facecolor="None", hatch="/", edgecolor="white")
        axes[1].axvspan(ratio * t0, ratio * t1, facecolor="None", hatch="/", edgecolor="white")
        axes[1].annotate(f"{_score(t_spans):.2f}", (ratio * t0, sample_rate * 0.51), annotation_clip=False)

        for span, char in zip(t_spans, chars):
            t0 = span.start * ratio
            axes[1].annotate(char, (t0, sample_rate * 0.55), annotation_clip=False)

    axes[1].set_xlabel("time [second]")
    axes[1].set_xlim([0, None])
    fig.tight_layout()

for idx, row in df.iterrows():
    audio_filepath = row['audio_filepath']
    transcript = row['text'].split()

    waveform, sample_rate = torchaudio.load(audio_filepath)

    with torch.inference_mode():
        emission, _ = model(waveform.to(device))
    star_dim = torch.zeros((1, emission.size(1), 1), device=emission.device, dtype=emission.dtype)
    emission = torch.cat((emission, star_dim), 2)

    emission = adjust_emission(emission, len(DICTIONARY))

    try:
        word_spans = compute_alignments(emission, transcript, DICTIONARY)
        plot_alignments(waveform, word_spans, emission, transcript)
        plt.show()
    except ValueError as e:
        print(f"Error processing row {idx}: {e}")
