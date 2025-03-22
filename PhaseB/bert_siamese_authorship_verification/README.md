Project Structure:

```bash
bert_siamese_authorship_verification/
│── config/                   # Configuration files
│   ├── config.yaml           # Hyperparameters and settings
│── data/                     # Folder for datasets (training, validation, test sets)
│   ├── processed/            # Processed dataset files
│   ├── raw/
│   │   ├── shakespeare/       # Folder containing Shakespeare's original works
│   │   │   ├── ...
│   │   ├── impostors/         # Folder containing impostors
│   │   │   ├── impostor_1/
│   │   │   │   ├── ...
│   │   │   ├── ...
│── logs/                     # Log files from training and evaluation
│── models/                   # Folder to store trained models and checkpoints
│   ├── bert_siamese.pth      # Saved model weights
│   ├── bert_siamese.py       # Model architecture implementation
│── notebooks/                # Jupyter notebooks for experimentation
│   ├── exploratory_analysis.ipynb
│   ├── model_testing.ipynb
│── src/                      # Source code for training and evaluation
│   ├── __init__.py           # Makes src a package
│   ├── clustering.py         # Anomaly detection and K-Medoids clustering
│   ├── data_loader.py        # Handles data preprocessing and loading
│   ├── dtw.py                # Dynamic Time Warping implementation
│   ├── evaluate.py           # Evaluation metrics and testing
│   ├── inference.py          # Script for making predictions
│   ├── preprocess.py         # Tokenization, chunking, and data preparation
│   ├── train.py              # Training loop for the model
│── tests/                    # Unit tests for the modules
│   ├── test_data_loader.py   # Tests for data loading
│   ├── test_model.py         # Tests for model architecture
│── main.py                   # Entry point for running the application
│── README.md                 # Project documentation
│── requirements.txt          # Python dependencies
```


Usage steps:
1) Put in data/raw/shakespeare the shakespeare dataset (txt files), and in data/raw/impostors put the impostors dataset (Look at the project architecture).
2) Install the required packages (pip install -r requirements.txt).
3) Preprocess the data - run the script to generate the dataset - python src/convert_txt_to_json.py
4) Train the model using python src/train.py. This will load the processed dataset, train the Siamese network and save it as models/bert_siamese.pth
5) Run Inference: To check authorship similarity between two text files, run: python main.py. It will print a similarity score between two texts.

