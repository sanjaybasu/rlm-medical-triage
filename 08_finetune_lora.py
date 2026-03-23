"""LoRA fine-tuning on RLM trajectories for post-training evaluation.

Fine-tunes base models on filtered correct RLM trajectories using unsloth
for efficient LoRA training. Outputs merged models in HuggingFace format
for subsequent GGUF conversion and ollama serving.

Prerequisites:
    pip install unsloth transformers datasets peft bitsandbytes

Usage:
    # Fine-tune Qwen3-8B on its own correct trajectories:
    python 08_finetune_lora.py --model qwen3:8b

    # Fine-tune with custom hyperparameters:
    python 08_finetune_lora.py --model qwen3:8b --epochs 5 --lr 1e-5 --rank 32

Output:
    output/finetuned/{model_slug}_lora/       (LoRA adapter)
    output/finetuned/{model_slug}_merged/     (merged full model)
"""
import json
import argparse
from pathlib import Path

TRAJECTORY_DIR = Path(__file__).parent / 'output' / 'trajectories'
OUTPUT_DIR = Path(__file__).parent / 'output' / 'finetuned'

# Mapping from ollama model names to HuggingFace model IDs
HF_MODEL_MAP = {
    'qwen3:8b': 'Qwen/Qwen3-8B',
    'llama3.1:8b': 'meta-llama/Llama-3.1-8B-Instruct',
    'qwen3:32b': 'Qwen/Qwen3-32B',
    'deepseek-r1:70b': 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B',
}


def load_training_data(model_slug: str):
    """Load filtered correct trajectories for a model."""
    path = TRAJECTORY_DIR / f"{model_slug}_train_filtered.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"No filtered trajectories found at {path}. "
            f"Run 07_generate_trajectories.py --model {model_slug.replace('_', ':')} first."
        )

    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))

    print(f"Loaded {len(examples)} training examples from {path}")
    return examples


def finetune(model_name: str, epochs: int = 3, lr: float = 2e-5,
             rank: int = 16, alpha: int = 32, batch_size: int = 4,
             grad_accum: int = 4, max_seq_length: int = 4096):
    """Run LoRA fine-tuning using unsloth."""
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        raise ImportError(
            "unsloth is required for fine-tuning. Install with: pip install unsloth"
        )
    from datasets import Dataset
    from transformers import TrainingArguments
    from trl import SFTTrainer

    model_slug = model_name.replace(":", "_")
    hf_model_id = HF_MODEL_MAP.get(model_name)
    if not hf_model_id:
        raise ValueError(f"Unknown model: {model_name}. Known: {list(HF_MODEL_MAP.keys())}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lora_dir = OUTPUT_DIR / f"{model_slug}_lora"
    merged_dir = OUTPUT_DIR / f"{model_slug}_merged"

    # Load training data
    examples = load_training_data(model_slug)
    if len(examples) < 10:
        print(f"WARNING: Only {len(examples)} training examples. Results may be poor.")

    # Format for SFTTrainer
    def format_example(example):
        """Convert messages list to a single formatted string using the model's native chat template."""
        return {"text": tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )}

    # Load model first so we have the tokenizer for chat template formatting
    print(f"Loading {hf_model_id}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=hf_model_id,
        max_seq_length=max_seq_length,
        dtype=None,  # auto-detect
        load_in_4bit=True,
    )

    dataset = Dataset.from_list([format_example(ex) for ex in examples])
    print(f"Dataset: {len(dataset)} examples")

    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=rank,
        lora_alpha=alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable_params:,} / {total_params:,} "
          f"({trainable_params/total_params*100:.2f}%)")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(lora_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
        seed=42,
        report_to="none",
    )

    # Train
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        args=training_args,
    )

    print(f"\nStarting training: {epochs} epochs, lr={lr}, rank={rank}")
    trainer.train()

    # Save LoRA adapter
    model.save_pretrained(str(lora_dir))
    tokenizer.save_pretrained(str(lora_dir))
    print(f"LoRA adapter saved to {lora_dir}")

    # Merge and save full model
    print("Merging LoRA weights into base model...")
    model = FastLanguageModel.for_inference(model)
    model.save_pretrained_merged(
        str(merged_dir),
        tokenizer,
        save_method="merged_16bit",
    )
    print(f"Merged model saved to {merged_dir}")

    # Save training config for reproducibility
    config = {
        "base_model": hf_model_id,
        "ollama_model": model_name,
        "training_examples": len(examples),
        "epochs": epochs,
        "learning_rate": lr,
        "lora_rank": rank,
        "lora_alpha": alpha,
        "batch_size": batch_size,
        "gradient_accumulation": grad_accum,
        "max_seq_length": max_seq_length,
        "trainable_params": trainable_params,
        "total_params": total_params,
    }
    with open(lora_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nFine-tuning complete for {model_name}.")
    print(f"Next steps:")
    print(f"  1. Convert to GGUF: python llama.cpp/convert_hf_to_gguf.py {merged_dir}")
    print(f"  2. Create ollama Modelfile pointing to the GGUF")
    print(f"  3. ollama create {model_slug}-rlm-ft -f Modelfile")
    print(f"  4. Run evaluation with the fine-tuned model")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', required=True,
                        help=f'Model name. Options: {list(HF_MODEL_MAP.keys())}')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--rank', type=int, default=16, help='LoRA rank')
    parser.add_argument('--alpha', type=int, default=32, help='LoRA alpha')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--grad-accum', type=int, default=4)
    parser.add_argument('--max-seq-length', type=int, default=4096)
    args = parser.parse_args()

    finetune(
        model_name=args.model,
        epochs=args.epochs,
        lr=args.lr,
        rank=args.rank,
        alpha=args.alpha,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_seq_length=args.max_seq_length,
    )


if __name__ == "__main__":
    main()
