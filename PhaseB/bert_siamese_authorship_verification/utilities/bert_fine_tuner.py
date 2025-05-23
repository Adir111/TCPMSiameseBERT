import os
import tensorflow as tf
from transformers import TFAutoModelForMaskedLM, BertTokenizer, DataCollatorForLanguageModeling
from huggingface_hub import HfApi
from datasets import Dataset


class BertFineTuner:
    def __init__(self, config, logger):
        self._config = config
        self._logger = logger
        self.tokenizer = BertTokenizer.from_pretrained(config['bert']['model'])
        self.model = TFAutoModelForMaskedLM.from_pretrained(config['bert']['model'])

    def finetune(self, texts):
        save_path = self._config['data']['fine_tuned_bert_model_path']

        dataset = Dataset.from_dict({"text": texts})
        self._logger.info(f"Loaded {len(dataset)} text segments for fine-tuning.")

        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                return_tensors="tf",
                max_length=self._config['bert']['maximum_sequence_length']
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

        api = HfApi(token=os.getenv("HF_TOKEN"))
        api.upload_folder(
            folder_path=save_path,
            repo_id=self._config['bert']['repository'],
            repo_type="model",
        )

        self._logger.info(f"Fine-tuned model uploaded to Hugging Face Hub: {self._config['bert']['repository']}.")


