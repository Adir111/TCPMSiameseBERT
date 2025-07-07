"""
Main CLI script to run various phases of the Siamese BERT authorship verification pipeline.
"""

import gc

from utilities import get_config, get_logger, convert_texts_to_json, convert_all_impostor_texts_to_json, BertFineTuner
from src.data_loader import DataLoader
from src.procedure import Procedure

# Load configuration and initialize logger and procedure
config = get_config()
logger = get_logger(config)
procedure = Procedure(config, logger)

# Load configuration and initialize logger and procedure
TESTED_COLLECTION_PATH = config['data']['shakespeare_path']
IMPOSTORS_PATH = config['data']['impostors_path']


def __handle_selection(func, selection='selection', extra_verification=False):
    """
    Confirm selection before running the provided function.
    Executes func() only if user confirms.

    Args:
        func (callable): Function to execute.
        selection (str): Description for prompt.
        extra_verification (bool): If True, requires additional user input 'verify'.

    Returns:
        Return value of func() or None.
    """
    confirm = input(f"Are you sure you want to run '{selection}'? (yes/no): ").strip().lower()
    if confirm == "yes" or confirm == "y":
        if extra_verification:
            verify = input(f"This selection requires extra verification, please write 'verify':").strip().lower()
            if verify != "verify":
                logger.log(f"Verification failed, cancelled '{selection}'")
                return None
        return func()
    else:
        logger.log(f"Cancelled '{selection}'.")
        return None


def __create_dataset():
    """Convert raw text files into JSON datasets for training and impostors."""
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
    """Run the training procedure."""
    try:
        logger.log("🚀 Starting Training Procedure...")
        procedure.run_training_procedure()

        logger.log("✅ Training Procedure completed!")
        logger.log({"status": "completed"})
    except Exception as e:
        logger.error(f"❌ {e}")


def __signal_generation():
    """Run the signal generation procedure."""
    try:
        logger.log("🚀 Starting Signal Generation Procedure...")
        procedure.run_signal_generation_procedure()

        logger.log("✅ Signal Generation Procedure completed!")
        logger.log({"status": "completed"})
    except Exception as e:
        logger.error(f"❌ {e}")


def __dtw():
    """Run the DTW procedure."""
    try:
        logger.log("🚀 Starting DTW Procedure...")
        procedure.run_distance_matrix_generation()

        logger.log("✅ DTW Procedure completed!")
        logger.log({"status": "completed"})
    except Exception as e:
        logger.error(f"❌ {e}")


def __isolation_forest():
    """Run the Isolation Forest procedure."""
    try:
        logger.log("🚀 Starting Isolation Forest Procedure...")
        procedure.run_isolation_forest_procedure()

        logger.log("✅ Isolation Forest Procedure completed!")
        logger.log({"status": "completed"})
    except Exception as e:
        logger.error(f"❌ {e}")


def __clustering():
    """Run the clustering procedure."""
    try:
        logger.log("🚀 Starting Clustering Procedure...")
        procedure.run_clustering_procedure()

        logger.log("✅ Clustering Procedure completed!")
        logger.log({"status": "completed"})
    except Exception as e:
        logger.error(f"❌ {e}")


def __fine_tune_berts():
    """Fine-tune BERT models for all impostors."""
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
    """Menu-driven CLI main loop."""
    selections = [
        ("Exit", None, False),
        ("Create Dataset", __create_dataset, False),
        ("Run Training Procedure", __training, False),
        ("Signal Generation Procedure", __signal_generation, False),
        ("DTW Procedure", __dtw, False),
        ("Isolation Forest Procedure", __isolation_forest, False),
        ("Clustering Procedure", __clustering, False),
        ("Fine-tune All BERTs", __fine_tune_berts, True)
    ]

    while True:
        logger.log("\n📜 Menu:")
        for index in range(len(selections)):
            logger.log(f"{index} - {selections[index][0]}")

        try:
            option = int(input("Select an option above: ").strip())
        except ValueError:
            logger.log("❌ Please enter a valid number.")
            continue

        # noinspection PyUnboundLocalVariable
        if option < 0 or option >= len(selections):
            logger.log("❌ Invalid option. Please try again.")
            continue
        elif selections[option][1] is None:
            logger.log("👋 Exiting. Have a great day!")
            break
        else:
            __handle_selection(selections[option][1], selections[option][0], selections[option][2])


if __name__ == "__main__":
    main()
