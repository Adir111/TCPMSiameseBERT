import gc

from PhaseB.bert_siamese_authorship_verification.utilities import BertFineTuner
from utilities import get_config, get_logger, convert_texts_to_json, convert_all_impostor_texts_to_json
from src.data_loader import DataLoader
from src.procedure import Procedure

# Load config
config = get_config()
logger = get_logger(config)
procedure = Procedure(config, logger)

# Paths
TESTED_COLLECTION_PATH = config['data']['shakespeare_path']
IMPOSTORS_PATH = config['data']['impostors_path']


def __handle_selection(func, selection='selection'):
    """
    Confirm selection before running the provided function.
    If the user types 'yes', the function is executed.
    """
    confirm = input(f"Are you sure you want to run '{selection}'? (yes/no): ").strip().lower()
    if confirm == "yes" or confirm == "y":
        return func()
    else:
        logger.log(f"Cancelled '{selection}'.")
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
        shakespeare_dir=TESTED_COLLECTION_PATH,
        impostor_dir=IMPOSTORS_PATH,
        shakespeare_collection_size=tested_collection_size,
        impostor_size=impostor_size
    )
    logger.log(f"✅ Dataset created")

    logger.log("🔄 Creating all impostors dataset...")
    convert_all_impostor_texts_to_json(impostor_dir=IMPOSTORS_PATH)
    logger.log(f"✅ All impostors dataset created")


def __training():
    """ Triggers the training script. """
    try:
        logger.log("🚀 Starting Training Procedure...")
        procedure.run_training_procedure()

        logger.log("✅ Training Procedure completed!")
        logger.log({"status": "completed"})
    except Exception as e:
        logger.error(f"❌ {e}")


def __classification():
    """ Triggers the classification procedure. """
    try:
        logger.log("🚀 Starting Classification Procedure...")
        procedure.run_classification_procedure()

        logger.log("✅ Classification Procedure completed!")
        logger.log({"status": "completed"})
    except Exception as e:
        logger.error(f"❌ {e}")


def __dtw():
    """ Triggers the DTW procedure. """
    try:
        logger.log("🚀 Starting DTW Procedure...")
        procedure.run_distance_matrix_generation()

        logger.log("✅ DTW Procedure completed!")
        logger.log({"status": "completed"})
    except Exception as e:
        logger.error(f"❌ {e}")


def __isolation_forest():
    """ Triggers the Isolation Forest procedure. """
    try:
        logger.log("🚀 Starting Isolation Forest Procedure...")
        procedure.run_isolation_forest_procedure()

        logger.log("✅ Isolation Forest Procedure completed!")
        logger.log({"status": "completed"})
    except Exception as e:
        logger.error(f"❌ {e}")


def __clustering():
    """ Triggers the clustering procedure. """
    try:
        logger.log("🚀 Starting Clustering Procedure...")
        procedure.run_clustering_procedure()

        logger.log("✅ Clustering Procedure completed!")
        logger.log({"status": "completed"})
    except Exception as e:
        logger.error(f"❌ {e}")


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
        logger.error("❌ No dataset found, please create one first!")
    except Exception as e:
        logger.error(f"❌ An error occurred during fine-tuning: {e}")


def main():
    """ Displays the menu and executes the selected action. """
    while True:
        logger.log("\n📜 Menu:")
        logger.log("1 - Create Dataset")
        logger.log("2 - Run Model Procedure")
        logger.log("3 - Classification Procedure")
        logger.log("4 - DTW Procedure")
        logger.log("5 - Isolation Forest Procedure")
        logger.log("6 - Clustering Procedure")
        logger.log("999 - Fine Tune All BERTs")
        logger.log("0 - Exit")

        option = input("Select an option above: ").strip()

        if option == "1":
            __handle_selection(__create_dataset, "Dataset Creation")
        elif option == "2":
            __handle_selection(__training, "Training Procedure")
        elif option == "3":
            __handle_selection(__classification, "Classification Procedure")
        elif option == "4":
            __handle_selection(__dtw, "DTW Procedure")
        elif option == "5":
            __handle_selection(__isolation_forest, "Isolation Forest Procedure")
        elif option == "6":
            __handle_selection(__clustering, "Clustering Procedure")
        elif option == "999":
            __handle_selection(__fine_tune_berts, "Fine Tune All BERTs")
        elif option == "0":
            logger.log("👋 Exiting. Have a great day!")
            break
        else:
            logger.log("❌ Invalid option. Please try again.")


if __name__ == "__main__":
    main()
