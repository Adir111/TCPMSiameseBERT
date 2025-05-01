from utilities import get_config, get_logger, convert_texts_to_json
from src_new.procedure import Procedure

# Load config
config = get_config()

# Paths
TESTED_COLLECTION_PATH = config['data']['shakespeare_path']
IMPOSTORS_PATH = config['data']['impostors_path']


def __create_dataset():
    """ Converts raw text files into a structured dataset in JSON format. """
    print("Create full dataset? (y/n)")
    full_dataset = input().strip().lower() == "y"
    impostor_size = None
    tested_collection_size = None

    if not full_dataset:
        while impostor_size is None or impostor_size < 2:
            print("Enter the number of impostors to include (at least 2):")
            impostor_size = int(input().strip())
        print("Enter the number of Shakespeare texts to include:")
        tested_collection_size = int(input().strip())

    print("🔄 Creating dataset...")
    convert_texts_to_json(
        config,
        shakespeare_dir=TESTED_COLLECTION_PATH,
        impostor_dir=IMPOSTORS_PATH,
        shakespeare_collection_size=tested_collection_size,
        impostor_size=impostor_size
    )
    print(f"✅ Dataset created")


def __train_model():
    """ Triggers the training script. """
    try:
        logger = get_logger(config)
        logger.log("🚀 Starting Full Procedure...")
        procedure = Procedure(config, logger)
        procedure.run()

        logger.log("✅ Procedure completed!")
        logger.log({"status": "completed"})
    except FileNotFoundError:
        print("❌ ERROR - no dataset found, please create one first!")


def main():
    """ Displays the menu and executes the selected action. """
    while True:
        print("\n📜 Menu:")
        print("1 - Create Dataset")
        print("2 - Run Model Procedure")
        print("3 - Exit")

        option = input("Select an option (1/2/3): ").strip()

        if option == "1":
            __create_dataset()
        elif option == "2":
            __train_model()
        elif option == "3":
            print("👋 Exiting. Have a great day!")
            break
        else:
            print("❌ Invalid option. Please try again.")


if __name__ == "__main__":
    main()
