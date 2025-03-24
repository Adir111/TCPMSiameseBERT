import os
from src.preprocess import TextPreprocessor
from src.inference import InferenceEngine
from bert_siamese_authorship_verification.config.get_config import get_config
from src.convert_txt_to_json import convert_texts_to_json
from src.train import train


# Load config
config = get_config()

# Paths
MODEL_PATH = config['data']['model_path']
SHAKESPEARE_PATH = config['data']['shakespeare_path']
IMPOSTORS_PATH = config['data']['impostors_path']
DATASET_PATH = config['data']['train_path']


def create_dataset():
    """ Converts raw text files into a structured dataset in JSON format. """
    print("🔄 Creating dataset...")
    convert_texts_to_json(SHAKESPEARE_PATH, IMPOSTORS_PATH, DATASET_PATH)
    print(f"✅ Dataset created at {DATASET_PATH}")


def train_model():
    """ Triggers the training script. """
    print("🚀 Starting training...")
    try:
        train()
        print("✅ Training completed!")
    except FileNotFoundError:
        print("❌ ERROR - no dataset found, please create one first!")


def compare_texts(file1, file2):
    """ Reads two text files and computes their similarity score. """
    with open(file1, "r", encoding="utf-8") as f1, open(file2, "r", encoding="utf-8") as f2:
        text1, text2 = f1.read(), f2.read()

    try:
        # Initialize model and preprocessor
        engine = InferenceEngine(model_path=MODEL_PATH)
        preprocessor = TextPreprocessor()
        similarity = engine.predict_similarity(text1, text2)
        print(f"\n📊 Similarity Score: {similarity:.4f}")
    except FileNotFoundError:
        print("❌ ERROR - no model found, please train one first!")


def select_directory(base_path):
    """ Lists directories (impostors) and lets the user select one. """
    print("\nAvailable authors:")
    authors = sorted(os.listdir(base_path))
    for i, author in enumerate(authors, 1):
        print(f"{i} - {author}")

    choice = input("Select an author by number: ").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(authors):
        return os.path.join(base_path, authors[int(choice) - 1])
    else:
        print("❌ Invalid selection. Try again.")
        return select_directory(base_path)


def select_file_from_directory(directory):
    """ Lists files in a directory and lets the user select one. """
    print("\nAvailable texts:")
    files = sorted(os.listdir(directory))
    for i, file in enumerate(files, 1):
        print(f"{i} - {file}")

    choice = input("Select a text file by number: ").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(files):
        return os.path.join(directory, files[int(choice) - 1])
    else:
        print("❌ Invalid selection. Try again.")
        return select_file_from_directory(directory)


def choose_text():
    """ Asks the user to select a text file from Shakespeare or impostors. """
    print("\nSelect text origin:")
    print("1 - Shakespeare")
    print("2 - Impostor")

    origin_choice = input("Enter your choice (1/2): ").strip()

    if origin_choice == "1":
        return select_file_from_directory(SHAKESPEARE_PATH)
    elif origin_choice == "2":
        impostor_dir = select_directory(IMPOSTORS_PATH)
        return select_file_from_directory(impostor_dir)
    else:
        print("❌ Invalid choice. Try again.")
        return choose_text()


def custom_text_selection():
    """ Allows the user to manually select two texts for similarity comparison. """
    print("\nChoose the first text:")
    text1 = choose_text()

    print("\nChoose the second text:")
    text2 = choose_text()

    compare_texts(text1, text2)


def get_text_similarity():
    """ Handles text similarity comparison between different texts. """
    print("\nChoose an option:")
    print("1 - Default: Shakespeare vs. Shakespeare")
    print("2 - Default: Shakespeare vs. Impostor")
    print("3 - Custom Text Selection")

    choice = input("Enter your choice (1/2/3): ").strip()

    if choice == "1":
        compare_texts(
            os.path.join(SHAKESPEARE_PATH, "A LOVERS COMPLAINT.txt"),
            os.path.join(SHAKESPEARE_PATH, "A MIDSUMMER NIGHT_S DREAM.txt"),
        )
    elif choice == "2":
        compare_texts(
            os.path.join(SHAKESPEARE_PATH, "A MIDSUMMER NIGHT_S DREAM.txt"),
            os.path.join(IMPOSTORS_PATH, "Benjamin Jonson/Discoveries Made Upon Men and Matter and Some Poems.txt"),
        )
    elif choice == "3":
        custom_text_selection()
    else:
        print("❌ Invalid option. Please try again.")


def main():
    """ Displays the menu and executes the selected action. """
    while True:
        print("\n📜 Menu:")
        print("1 - Create Dataset")
        print("2 - Train Model")
        print("3 - Get Text Similarity")
        print("4 - Exit")

        option = input("Select an option (1/2/3/4): ").strip()

        if option == "1":
            create_dataset()
        elif option == "2":
            train_model()
        elif option == "3":
            get_text_similarity()
        elif option == "4":
            print("👋 Exiting. Have a great day!")
            break
        else:
            print("❌ Invalid option. Please try again.")


if __name__ == "__main__":
    main()
