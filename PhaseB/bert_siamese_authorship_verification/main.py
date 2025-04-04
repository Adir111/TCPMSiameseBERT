import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.get_config import get_config
from src.convert_txt_to_json import convert_texts_to_json
from src.train import full_procedure

# Load config
config = get_config()

# Paths
TESTED_COLLECTION_PATH = config['data']['shakespeare_path']
IMPOSTORS_PATH = config['data']['impostors_path']


def create_dataset():
    """ Converts raw text files into a structured dataset in JSON format. """
    print("Create full dataset? (y/n)")
    full_dataset = input().strip().lower() == "y"
    impostor_size = None
    tested_collection_size = None

    if not full_dataset:
        print("Enter the number of impostor texts to include:")
        impostor_size = int(input().strip())
        print("Enter the number of Shakespeare texts to include:")
        tested_collection_size = int(input().strip())

    print("🔄 Creating dataset...")
    convert_texts_to_json(TESTED_COLLECTION_PATH, IMPOSTORS_PATH, tested_collection_size=tested_collection_size, impostor_size=impostor_size)
    print(f"✅ Dataset created")


def train_model():
    """ Triggers the training script. """
    print("🚀 Starting training...")
    try:
        full_procedure()
        print("✅ Training completed!")
    except FileNotFoundError:
        print("❌ ERROR - no dataset found, please create one first!")


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
        return select_file_from_directory(TESTED_COLLECTION_PATH)
    elif origin_choice == "2":
        impostor_dir = select_directory(IMPOSTORS_PATH)
        return select_file_from_directory(impostor_dir)
    else:
        print("❌ Invalid choice. Try again.")
        return choose_text()


def main():
    """ Displays the menu and executes the selected action. """
    while True:
        print("\n📜 Menu:")
        print("1 - Create Dataset")
        print("2 - Run Model Procedure")
        print("3 - Exit")

        option = input("Select an option (1/2/3): ").strip()

        if option == "1":
            create_dataset()
        elif option == "2":
            train_model()
        elif option == "3":
            print("👋 Exiting. Have a great day!")
            break
        else:
            print("❌ Invalid option. Please try again.")


if __name__ == "__main__":
    main()
