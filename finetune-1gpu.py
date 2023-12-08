import os
import sys
from typing import List

import fire
import torch
import transformers
from datasets import load_dataset

from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

import bitsandbytes as bnb

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
    set_peft_model_state_dict
)

from transformers import LlamaForCausalLM, LlamaTokenizer

from utils.prompter import Prompter


class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        kwargs["model"].save_pretrained(checkpoint_folder)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        torch.save({}, pytorch_model_path)
        return control


class LoadBestPeftModelCallback(TrainerCallback):
    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        print(f"Loading best peft model from {state.best_model_checkpoint} (score: {state.best_metric}).")
        best_model_path = os.path.join(state.best_model_checkpoint, "adapter_model.bin")
        adapters_weights = torch.load(best_model_path)
        model = kwargs["model"]
        set_peft_model_state_dict(model, adapters_weights)
        return control

def load_tokenizer(base_model):
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    print("pre-trained model's BOS EOS and PAD token id:",bos,eos,pad," => It should be 1 2 None")

    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "right"
    return tokenizer

def load_data(data_path,
              tokenizer,
              cutoff_len,
              prompt_template_name='alpaca', 
              train_on_inputs=False, 
              add_eos_token=False, 
              val_set_size=0):
    prompter = Prompter(prompt_template_name)
        
    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)
        
    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"])
        
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"])
            
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token)
            
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # TODO: Speed up?
        return tokenized_full_prompt
    
    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle(seed=42).map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle(seed=42).map(generate_and_tokenize_prompt)
        )
    else:
        train_data = data["train"].shuffle(seed=42).map(generate_and_tokenize_prompt)
        val_data = None
    
    return train_data, val_data


def select_samples_by_len(dataset, min_length, max_length):
    def length_filter(sample, min_length, max_length):
        return min_length <= len(sample['input_ids']) <= max_length

    filtered_dataset = dataset.filter(lambda sample: length_filter(sample, min_length, max_length))
    return filtered_dataset


def forward_hook_fn(module, input, output):
    # Calculate the memory size of the output activation
    # Assuming the output is a torch Tensor and using float32 (4 bytes per element)
    batch_size, seq_length, feature_size = output.size()
    memory_size = batch_size * seq_length * feature_size * 4  # in bytes
    print(f"Act.Mem Profile: Layer: {module.__class__.__name__}, Output Size: {output.size()}, Memory: {memory_size} bytes, \
          Output Data Type: {output.dtype}")

def pack_hook(tensor):
    tensor_size = tensor.nelement() * tensor.element_size()
    print(f"Saving tensor of size: {tensor_size} bytes")
    return tensor  # You can return a handle for the unpack hook
    # return None

def unpack_hook(tensor):
    # input("Before Train, Press Enter to continue...")
    if tensor is not None:
        tensor_size = tensor.nelement() * tensor.element_size()
        print(f"Unpacking tensor of size: {tensor_size} bytes")
    return tensor

def train(
    # model/data params
    base_model: str = "", 
    data_path: str = "",
    output_dir: str = "",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 8,
    num_epochs: int = 1,
    learning_rate: float = 3e-4,
    cutoff_len: int = 4096,
    val_set_size: int = 0,
    lr_scheduler: str = "cosine",
    warmup_steps: int = 100, 
    # lora hyperparams
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    # from peft docs: ["q_proj", "k_proj", "v_proj", "o_proj", "fc_in", "fc_out", "wte", "gate_proj", "down_proj", "up_proj"]
    lora_target_modules: List[str] = ["gate_proj", "down_proj", "up_proj"],
    # llm hyperparams
    train_on_inputs: bool = False,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca",
    gradient_checkpointing: bool = False,
    seq_min: int = 0,
    seq_max: int = 4096,
    debug: bool = False,
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Params using prompt template {prompt_template_name}:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lr_scheduler: {lr_scheduler}\n"
            f"warmup_steps: {warmup_steps}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"gradient_checkpointing: {gradient_checkpointing}\n"
            f"seq_min: {seq_min}\n"
            f"seq_max: {seq_max}\n"
            f"debug: {debug}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    #device_map = "auto"
    device_map = {"":0}
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    print("ddp is: ", ddp)
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
        print("gradient_accumulation_steps: ", gradient_accumulation_steps)
    
    print("gradient_accumulation_steps: ", gradient_accumulation_steps)
    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )

    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    tokenizer = load_tokenizer(base_model)
    train_data, val_data = load_data(data_path, tokenizer, cutoff_len, 
                                     prompt_template_name, train_on_inputs, add_eos_token, val_set_size)    
        
    train_data = select_samples_by_len(train_data,seq_min,seq_max)
    
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map)


    model = prepare_model_for_int8_training(model, use_gradient_checkpointing=False)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM")

    model = get_peft_model(model, config)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()

    print(model)
    
    ################## debug ################
    
    if debug:
    
        for layer in model.modules(): 
            # if isinstance(layer, (torch.nn.MultiheadAttention, torch.nn.Linear)):
            if isinstance(layer, (bnb.nn.Linear8bitLt, torch.nn.Linear)):
                print(f"registering hook to {layer.__class__.__name__}")
                layer.register_forward_hook(forward_hook_fn)
          
              
        # batch_size=1
        # seq_len = 512
        # input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len)).to('xpu:0')  # Example input
        # # # with torch.no_grad():
        # # with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        # #     output = model(input_ids)
        # #     output.backward()
        input("Before Train, Press Enter to continue...")
        
    # print("device count: ", torch.cuda.device_count())
    #if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
    #    model.is_parallelizable = True
    #    model.model_parallel = True

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            gradient_checkpointing=gradient_checkpointing,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            # dataloader_num_workers=16,
            bf16=True,
            #fp16=True,
            logging_steps=1,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if val_set_size > 0 else None,
            save_steps=1000,
            lr_scheduler_type=lr_scheduler,
            output_dir=output_dir,
            save_total_limit=2,
            load_best_model_at_end=True if val_set_size > 0 else False,
            #ddp_find_unused_parameters=False if ddp else None,
            ddp_find_unused_parameters=False,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        # callbacks=[SavePeftModelCallback, LoadBestPeftModelCallback], # ONLY USE LoadBestPeftModelCallback if val_set_size > 0
    )
    model.config.use_cache = False

    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)

    if debug:
        with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)
    # model.base_model.save_pretrained(output_dir)
    pytorch_model_path = os.path.join(output_dir, "pytorch_model.bin")
    torch.save({}, pytorch_model_path)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    torch.cuda.empty_cache() 
    fire.Fire(train)
