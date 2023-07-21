from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM, 
    BitsAndBytesConfig, 
    HfArgumentParser, 
    TrainingArguments, 
    set_seed)

from llm.finetuner.sft_trainer import SFTTrainer
from llm.finetuner.utils import print_trainable_parameters
from llm.finetuner.peft_helper import get_accelerate_model

tqdm.pandas()

# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with SFTTrainer
    """

    model_name: Optional[str] = field(default="facebook/opt-350m", metadata={"help": "the model name"})
    output_dir: str = field(default='./output', metadata={"help": 'The output dir for logs and checkpoints'})
    dataset_name: Optional[str] = field(default="timdettmers/openassistant-guanaco", metadata={"help": "the dataset name"})
    dataset_text_field: Optional[str] = field(default="text", metadata={"help": "the text field of the dataset"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    batch_size: Optional[int] = field(default=64, metadata={"help": "the batch size"})
    max_seq_length: Optional[int] = field(default=512, metadata={"help": "Max length of sequences"})
    gradient_accumulation_steps: Optional[int] = field(default=16, metadata={"help": "the number of gradient accumulation steps"})
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
    use_peft: Optional[bool] = field(default=False, metadata={"help": "Wether to use PEFT or not to train adapters"})
    trust_remote_code: Optional[bool] = field(default=True, metadata={"help": "Enable `trust_remote_code`"})
    output_dir: Optional[str] = field(default="output", metadata={"help": "the output directory"})
    peft_lora_r: Optional[int] = field(default=64, metadata={"help": "the r parameter of the LoRA adapters"})
    peft_lora_alpha: Optional[int] = field(default=16, metadata={"help": "the alpha parameter of the LoRA adapters"})
    logging_steps: Optional[int] = field(default=1, metadata={"help": "the number of logging steps"})
    use_auth_token: Optional[bool] = field(default=True, metadata={"help": "Use HF auth token to access the model"})
    gradient_checkpointing: bool = field(default=True, metadata={"help": 'Use gradient checkpointing. You want to use this.'})
    double_quant: bool = field(default=True, metadata={"help": "Compress the quantization statistics through double quantization."})
    report_to: str = field(default='none', metadata={"help": "To use wandb or something else for reporting."})
    adam8bit: bool = field(default=False, metadata={"help": "Use 8-bit adam."})
    lora_dropout: float = field(default=0.0, metadata={"help":"Lora dropout."})


def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    if script_args.use_peft:
        model, peft_config = get_accelerate_model(script_args)

    dataset = load_dataset(script_args.dataset_name, split="train")

    model.config.use_cache = False
    print('loaded model')

    print_trainable_parameters(model)
    set_seed(script_args.seed)

    training_args = TrainingArguments(
    output_dir=script_args.output_dir,
    per_device_train_batch_size=script_args.batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    learning_rate=script_args.learning_rate,
    logging_steps=script_args.logging_steps,
    )

    trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    dataset_text_field=script_args.dataset_text_field,
    peft_config=peft_config)

    trainer.train()