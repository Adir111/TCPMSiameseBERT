Project Structure:

```bash
bert_siamese_authorship_verification/
│── config/                   # Configuration files
│   ├── config.yaml           # Hyperparameters and settings
│── data/                     # Folder for datasets (training, validation, test sets)
│   ├── processed/            # Processed dataset files
│   ├── raw/
│   │   ├── shakespeare/       # Folder containing Shakespeare's original works and some unknown ones
│   │   │   ├── ...
│   │   ├── impostors/         # Folder containing impostors
│   │   │   ├── impostor_1/
│   │   │   │   ├── ...
│   │   │   ├── ...
│── models/                   # Folder to store trained models and checkpoints
│   ├── bert_siamese.py       # Model architecture implementation
│── src/                      # Source code for training and evaluation
│   ├── __init__.py           # Makes src a package
│   ├── clustering.py         # Anomaly detection and K-Medoids clustering
│   ├── data_loader.py        # Handles data preprocessing and loading
│   ├── dtw.py                # Fast Dynamic Time Warping implementation
│   ├── evaluate.py           # Evaluation metrics and testing
│   ├── inference.py          # Script for making predictions
│   ├── preprocess.py         # Tokenization, chunking, and data preparation
│   ├── train.py              # Model training and inference procedure
├── tests/                # ✅ All unit and integration tests
│   ├── test_anomaly.py
│   ├── test_data_loader.py
│   ├── test_dtw.py
│   ├── test_model.py
│   ├── test_pipeline.py
│   └── test_preprocess.py
│── main.py                   # Entry point for running the application
│── README.md                 # Project documentation
```


Usage steps:
1) Put in data/raw/shakespeare the shakespeare dataset (txt files), and in data/raw/impostors put the impostors dataset (Look at the project architecture).
2) Install the required packages (pip install -r requirements.txt).
3) Run main.py and use the menu according to the instructions. Or:
   1) Preprocess the data - run the script to generate the dataset - python src/convert_txt_to_json.py
   2) Train the model using python src/train.py. This will load the processed dataset, train the Siamese network and use it for inference on the Shakespearian texts.
   3) Result from previous step is a T-SNE projection of the results.

