from src.inference import InferenceEngine

if __name__ == "__main__":
    engine = InferenceEngine(model_path="models/bert_siamese.pth")
    text1 = "text name.txt"
    text2 = "text name.txt"

    similarity = engine.predict_similarity(text1, text2)
    print(f"Predicted Similarity Score: {similarity}")
