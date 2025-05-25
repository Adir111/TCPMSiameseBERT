from pathlib import Path

from utilities import get_config, get_logger, convert_texts_to_json, convert_all_impostor_texts_to_json
from src.procedure import Procedure

# Load config
config = get_config()
logger = get_logger(config)

# Paths
TESTED_COLLECTION_PATH = config['data']['shakespeare_path']
IMPOSTORS_PATH = config['data']['impostors_path']


def __create_dataset():
    """ Converts raw text files into a structured dataset in JSON format. """
    logger.log("Create full dataset? (y/n)")
    full_dataset = input().strip().lower() == "y"
    impostor_size = None
    tested_collection_size = None

    if not full_dataset:
        while impostor_size is None or impostor_size < 2:
            logger.log("Enter the number of impostors to include (at least 2):")
            impostor_size = int(input().strip())
        logger.log("Enter the number of Shakespeare texts to include:")
        tested_collection_size = int(input().strip())

    logger.log("🔄 Creating dataset...")
    convert_texts_to_json(
        config,
        shakespeare_dir=TESTED_COLLECTION_PATH,
        impostor_dir=IMPOSTORS_PATH,
        shakespeare_collection_size=tested_collection_size,
        impostor_size=impostor_size
    )
    logger.log(f"✅ Dataset created")

    logger.log("🔄 Creating all impostors dataset...")
    convert_all_impostor_texts_to_json(config, impostor_dir=IMPOSTORS_PATH)
    logger.log(f"✅ All impostors dataset created")


def __train_model():
    """ Triggers the training script. """
    try:
        starting_iteration = None
        while starting_iteration is None or starting_iteration < 0:
            try:
                starting_iteration = int(input("Enter the starting iteration (0 for new model): ").strip())
            except ValueError:
                logger.log("❌ Invalid input. Please enter a valid integer.")
        logger.log("🚀 Starting Full Procedure...")
        procedure = Procedure(config, logger)
        procedure.run(starting_iteration)

        logger.log("✅ Procedure completed!")
        logger.log({"status": "completed"})
    except FileNotFoundError:
        logger.log("❌ ERROR - no dataset found, please create one first!")


def main():
    """ Displays the menu and executes the selected action. """
    while True:
        logger.log("\n📜 Menu:")
        logger.log("1 - Create Dataset")
        logger.log("2 - Run Model Procedure")
        logger.log("3 - Exit")

        option = input("Select an option (1/2/3): ").strip()

        if option == "1":
            __create_dataset()
        elif option == "2":
            __train_model()
        elif option == "3":
            logger.log("👋 Exiting. Have a great day!")
            break
        else:
            logger.log("❌ Invalid option. Please try again.")


if __name__ == "__main__":
    main()
