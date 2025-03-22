import torch
from src.preprocess import TextPreprocessor
from src.inference import InferenceEngine

if __name__ == "__main__":
    engine = InferenceEngine(model_path="models/bert_siamese.pth")
    preprocessor = TextPreprocessor()

    # Shakespeare vs. Shakespeare
    with open("data/raw/shakespeare/A LOVERS COMPLAINT.txt", "r", encoding="utf-8") as f:
        text1 = f.read()

    with open("data/raw/shakespeare/A MIDSUMMER NIGHT_S DREAM.txt", "r", encoding="utf-8") as f:
        text2 = f.read()

    similarity = engine.predict_similarity(text1, text2)
    print(f"Shakespeare vs. Shakespeare Similarity Score: {similarity:.4f}")

    # Shakespeare vs. Impostor
    with open("data/raw/shakespeare/A MIDSUMMER NIGHT_S DREAM.txt", "r", encoding="utf-8") as f:
        text1 = f.read()

    with open("data/raw/impostors/Benjamin Jonson/Discoveries Made Upon Men and Matter and Some Poems.txt",
              "r", encoding="utf-8") as f:
        text2 = f.read()

    similarity = engine.predict_similarity(text1, text2)
    print(f"Shakespeare vs. Impostor Similarity Score: {similarity:.4f}")
