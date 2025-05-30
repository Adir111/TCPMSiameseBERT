import gc
from pathlib import Path

from PhaseB.bert_siamese_authorship_verification.utilities import BertFineTuner
from utilities import get_config, get_logger, convert_texts_to_json, convert_all_impostor_texts_to_json
from src.data_loader import DataLoader
from src.procedure import Procedure

# Load config
config = get_config()
logger = get_logger(config)

# Paths
TESTED_COLLECTION_PATH = config['data']['shakespeare_path']
IMPOSTORS_PATH = config['data']['impostors_path']


def __handle_selection(func):
    """
    Confirm selection before running the provided function.
    If the user types 'yes', the function is executed.
    """
    confirm = input(f"Are you sure you want to run '{func.__name__}'? (yes/no): ").strip().lower()
    if confirm == "yes" or confirm == "y":
        return func()
    else:
        print(f"Cancelled '{func.__name__}'.")
        return None


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


def __fine_tune_berts():
    """ Fine-tunes BERT models for the Siamese architecture. """
    try:
        data_loader = DataLoader(config)
        all_impostors = data_loader.get_all_impostors_data()
        logger.log("🔄 Fine-tuning BERT models...")
        for impostor in all_impostors:
            bert_fine_tuner = BertFineTuner(config, logger)
            bert_fine_tuner.finetune(impostor)

            del bert_fine_tuner  # Free memory after each fine-tuning
            gc.collect()
        logger.log("✅ BERT models fine-tuned successfully!")
    except FileNotFoundError:
        logger.log("❌ ERROR - no dataset found, please create one first!")
    except Exception as e:
        logger.log(f"❌ An error occurred during fine-tuning: {e}")


def __run_procedure():
    """ Triggers the training script. """
    try:
        logger.log("🚀 Starting Full Procedure...")
        procedure = Procedure(config, logger)
        procedure.run()

        logger.log("✅ Procedure completed!")
        logger.log({"status": "completed"})
    except Exception as e:
        logger.log(f"❌ ERROR - {e}")


def main():
    """ Displays the menu and executes the selected action. """
    while True:
        logger.log("\n📜 Menu:")
        logger.log("1 - Create Dataset")
        logger.log("2 - Run Model Procedure")
        logger.log("999 - Fine Tune All BERTs")
        logger.log("3 - Exit")

        option = input("Select an option (1/2/3/999): ").strip()

        if option == "1":
            __handle_selection(__create_dataset)
        elif option == "2":
            __handle_selection(__run_procedure)
            break
        elif option == "999":
            __handle_selection(__fine_tune_berts)
            break
        elif option == "3":
            logger.log("👋 Exiting. Have a great day!")
            break
        else:
            logger.log("❌ Invalid option. Please try again.")


if __name__ == "__main__":
    main()
