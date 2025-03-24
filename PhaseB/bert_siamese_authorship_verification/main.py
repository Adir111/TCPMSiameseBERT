import os

from src.preprocess import TextPreprocessor
from src.inference import InferenceEngine
from bert_siamese_authorship_verification.config.get_config import get_config

# Load config
config = get_config()

if __name__ == "__main__":
    # Load paths from config
    model_path = config['data']['model_path']
    shakespeare_path = config['data']['shakespeare_path']
    impostors_path = config['data']['impostors_path']

    # Initialize model & preprocessor
    engine = InferenceEngine(model_path=model_path)
    preprocessor = TextPreprocessor()

    # Shakespeare vs. Shakespeare
    with open(os.path.join(shakespeare_path, "A LOVERS COMPLAINT.txt"), "r", encoding="utf-8") as f:
        text1 = f.read()

    with open(os.path.join(shakespeare_path, "A MIDSUMMER NIGHT_S DREAM.txt"), "r", encoding="utf-8") as f:
        text2 = f.read()

    similarity = engine.predict_similarity(text1, text2)
    print(f"Shakespeare vs. Shakespeare Similarity Score: {similarity:.4f}")

    # Shakespeare vs. Impostor
    with open(os.path.join(shakespeare_path, "A MIDSUMMER NIGHT_S DREAM.txt"), "r", encoding="utf-8") as f:
        text1 = f.read()

    with open(os.path.join(impostors_path, "Benjamin Jonson/Discoveries Made Upon Men and Matter and Some Poems.txt"),
              "r", encoding="utf-8") as f:
        text2 = f.read()

    similarity = engine.predict_similarity(text1, text2)
    print(f"Shakespeare vs. Impostor Similarity Score: {similarity:.4f}")
