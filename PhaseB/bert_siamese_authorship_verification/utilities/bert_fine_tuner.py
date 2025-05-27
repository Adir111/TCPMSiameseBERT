import tensorflow as tf
from transformers import TFAutoModelForMaskedLM, BertTokenizer, DataCollatorForLanguageModeling
from huggingface_hub import HfApi, login
from datasets import Dataset
from pathlib import Path


class BertFineTuner:
    def __init__(self, config, logger):
        self._config = config
        self._logger = logger
        self.tokenizer = BertTokenizer.from_pretrained(config['bert']['model'])
        self.model = TFAutoModelForMaskedLM.from_pretrained(config['bert']['model'])

    def finetune(self, impostor):
        impostor_name = impostor['author']
        impostor_texts = impostor['texts']
        save_path = Path(self._config['data']['fine_tuned_bert_model_path']) / f"{impostor_name}/"

        dataset = Dataset.from_dict({"text": impostor_texts})
        self._logger.info(f"Loaded {len(dataset)} text segments for fine-tuning for author: {impostor_name}.")

        def tokenize_function(data):
            return self.tokenizer(
                data["text"],
                truncation=True,
                padding="max_length",
                return_tensors="tf",
                max_length=self._config['bert']['max_sequence_length']
            )

        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        self._logger.info("Tokenization completed.")

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=self._config['bert']['mlm_probability'],
            return_tensors="tf"
        )
        self._logger.info("Data collator created.")

        # Training Setup
        tf_dataset = tokenized_dataset.to_tf_dataset(
            columns=["input_ids", "attention_mask"],
            label_cols=["input_ids"],  # MLM target is input_ids again
            shuffle=True,
            batch_size=self._config['bert']['train_batch_size'],
            collate_fn=data_collator
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
        self.model.compile(optimizer=optimizer)

        self.model.fit(tf_dataset, epochs=self._config['bert']['num_epochs'])

        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

        self._logger.info(f"Fine-tuned model saved to {save_path}.")

        hf_token = self._config['bert']['token']
        login(token=hf_token)
        api = HfApi(token=hf_token)
        api.upload_folder(
            folder_path=save_path,
            repo_id=self._config['bert']['repository'],
            repo_type="model",
            path_in_repo=impostor_name
        )

        self._logger.info(f"Fine-tuned model uploaded to Hugging Face Hub: {self._config['bert']['repository']}.")


