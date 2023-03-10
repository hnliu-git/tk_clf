from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

from utils import vac_tags_dict, vac_main_dict, tokenize_and_align_labels, tokenize_and_align_labels_and_chunk, TKDataset 
from metric_utils.metrics import initialize_metrics
from metric_utils.data_utils import TagDict

import datasets

import numpy as np
import wandb
# import os
# os.environ["WANDB_DISABLED"] = "true"

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [p for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [l for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metrics.compute(predictions=true_predictions, gold=true_labels)

    return results

epoch = 20
bsz = 16
data_folder = 'data/vac-nl-orig/'
model_name = 'xlm-roberta-base'
exp_name = '%s-nl-epoch%d'%(model_name, epoch)

data_files = {
    'train': data_folder + 'train.jsonl',
    'validation': data_folder + 'devel.jsonl',
    'test': data_folder + 'test.jsonl'
}

dataset = datasets.load_dataset('json', data_files=data_files)
dataset.cleanup_cache_files()

print(f'train: {len(dataset["train"])}')
print(f'eval: {len(dataset["validation"])}')
print(f'test: {len(dataset["test"])}')

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenized_dataset = dataset.map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=True)

data_collator = DataCollatorForTokenClassification(tokenizer)

model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=len(vac_tags_dict)
)

metrics = initialize_metrics(
    metric_names=['accuracy', 'entity_score', 'entity_overlap'],
    tags=TagDict.from_file('metric_utils/vac-phrases-full-tags.txt'),
    main_entities=open('metric_utils/main_ents.txt').read().splitlines()
)

wandb.init(project="bert_vac_ner", name=exp_name)

training_args = TrainingArguments(
    output_dir="./fine_tune_bert_output",
    evaluation_strategy="steps",
    learning_rate=2e-5,
    per_device_train_batch_size=bsz,
    per_device_eval_batch_size=bsz,
    num_train_epochs=epoch,
    weight_decay=0.01,
    logging_steps = 100,
    report_to="wandb",
    run_name = "ep_01_tokenized_02",
    save_strategy='no'
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation'],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
wandb.finish()
