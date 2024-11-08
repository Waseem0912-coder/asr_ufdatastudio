import pandas as pd
from montreal_forced_aligner.g2p.generator import PyniniGenerator
from montreal_forced_aligner.models import G2PModel, ModelManager

# Set language and model path
language = "english_india_mfa"

# If you haven't downloaded the model
# manager = ModelManager()
# manager.download_model("g2p", language)

model_path = G2PModel.get_pretrained_path(language)

# Instantiate MFA G2P model
g2p = PyniniGenerator(g2p_model_path=model_path, num_pronunciations=1)
g2p.setup()

# Assume 'data' is the dataframe containing the transcripts in the 'text' column
data = pd.DataFrame({
    'text': ["hello world", "how are you", "this is a test"]
})

# Function to convert text to phonetic representation
def convert_to_phonetic(text):
    return ' '.join(g2p.generate(text))

# Apply phonetic conversion and add it as a new column
data['phonetic'] = data['text'].apply(convert_to_phonetic)

# Save the dataframe to CSV
data.to_csv('transcripts_with_phonetics.csv', index=False)

print("Phonetic conversion completed and saved to 'transcripts_with_phonetics.csv'")
