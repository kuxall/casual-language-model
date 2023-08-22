import math
from transformers import Trainer, TrainingArguments
from transformers import AutoConfig, AutoModelForCausalLM
from transformers import AutoTokenizer
from datasets import load_dataset

datasets = load_dataset(
    "text", data_files={"train": "nepali_c2.txt", "test": "nepali_c4.txt"})

# breakpoint()
model_checkpoint = "gpt2"
tokenizer_checkpoint = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)


def tokenize_function(examples):
    return tokenizer(examples["text"])


tokenized_datasets = datasets.map(
    tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

block_size = 128


def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=1000,
    num_proc=4,
)

config = AutoConfig.from_pretrained(model_checkpoint)
model = AutoModelForCausalLM.from_config(config)


model_directory = "nepGPTmodel"
tokenizer_directory = "nepGPTtokenizer"

# Training arguments
training_args = TrainingArguments(
    output_dir=model_directory,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    num_train_epochs=5,
    push_to_hub=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["test"],
)

trainer.train()

eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

model.save_pretrained(model_directory)
tokenizer.save_pretrained(tokenizer_directory)

trainer.push_to_hub()
