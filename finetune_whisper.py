"""
MC Hub AI — Whisper Fine-Tune (STT) — GPU Accelerated
=======================================================
Compatible: transformers >= 4.x, Python 3.11+

Input:  training_data/stt_metadata.txt
Output: models/whisper-vi-finetuned/

GPU: RTX 4060, FP16, gradient checkpointing
"""

import os
import sys
import json
import warnings
import torch
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Union

warnings.filterwarnings("ignore")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ----------------------------------------------------------------
#  Config
# ----------------------------------------------------------------
MODEL_NAME       = "openai/whisper-small"
STT_METADATA     = Path("./training_data/stt_metadata.txt")
OUTPUT_DIR       = Path("./models/whisper-vi-finetuned")
LANGUAGE         = "vi"
TASK             = "transcribe"
MAX_STEPS        = 1000         # ~30 min on RTX 4060 with 1999 samples
BATCH_SIZE       = 4            # Safe for 8GB VRAM with whisper-small
GRAD_ACCUM       = 4            # Effective batch = 16
LEARNING_RATE    = 1e-5
WARMUP_STEPS     = 50
SAVE_STEPS       = 200
EVAL_STEPS       = 200
LOG_STEPS        = 20

# ----------------------------------------------------------------
#  GPU check
# ----------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n{'='*50}")
print(f" Whisper Fine-Tune — Vietnamese MC Voice")
print(f"{'='*50}")
print(f" Device : {device.upper()}")
if device == "cuda":
    print(f" GPU    : {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f" VRAM   : {vram:.1f} GB")
print(f" Model  : {MODEL_NAME}")
print(f" Steps  : {MAX_STEPS}")
print(f"{'='*50}\n")


# ----------------------------------------------------------------
#  Load metadata
# ----------------------------------------------------------------
def load_metadata():
    if not STT_METADATA.exists():
        raise FileNotFoundError(f"{STT_METADATA} not found. Run preprocess_audio.py first.")

    samples = []
    for line in STT_METADATA.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        parts = line.split("|", 1)
        if len(parts) == 2:
            wav_path, text = parts
            if Path(wav_path).exists():
                samples.append({"audio": wav_path, "text": text.strip()})

    print(f"[Data] {len(samples)} valid samples loaded")

    split = int(len(samples) * 0.9)
    return samples[:split], samples[split:]


# ----------------------------------------------------------------
#  Prepare features for Whisper
# ----------------------------------------------------------------
def make_features(samples, processor, max_samples=None):
    """Convert WAV files → Whisper input features + token labels."""
    processed = []
    skipped = 0

    if max_samples:
        samples = samples[:max_samples]

    for s in samples:
        try:
            audio, _ = librosa.load(s["audio"], sr=16000, mono=True)

            # Whisper max 30s input
            if len(audio) > 16000 * 30:
                audio = audio[:16000 * 30]

            feat = processor.feature_extractor(
                audio, sampling_rate=16000, return_tensors="np"
            )
            label_ids = processor.tokenizer(s["text"]).input_ids

            processed.append({
                "input_features": feat.input_features[0],
                "labels": label_ids
            })
        except Exception:
            skipped += 1

    print(f"[Data] Prepared: {len(processed)} | Skipped: {skipped}")
    return processed


# ----------------------------------------------------------------
#  Data collator
# ----------------------------------------------------------------
@dataclass
class WhisperDataCollator:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features):
        inputs = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(inputs, return_tensors="pt")

        label_feat = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_feat, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# ----------------------------------------------------------------
#  Fine-tune
# ----------------------------------------------------------------
def finetune():
    from transformers import (
        WhisperProcessor,
        WhisperForConditionalGeneration,
        Seq2SeqTrainingArguments,
        Seq2SeqTrainer,
    )
    import evaluate

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load processor + model
    print("[Model] Loading Whisper processor...")
    processor = WhisperProcessor.from_pretrained(
        MODEL_NAME, language=LANGUAGE, task=TASK
    )

    print("[Model] Loading Whisper model weights...")
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False

    if device == "cuda":
        model = model.to("cuda")

    # Load + prepare data
    train_raw, eval_raw = load_metadata()
    print("[Data] Extracting features...")
    train_data = make_features(train_raw, processor)
    eval_data  = make_features(eval_raw, processor)

    data_collator = WhisperDataCollator(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # WER metric
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids  = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str  = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": wer_metric.compute(predictions=pred_str, references=label_str)}

    # Training args
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        max_steps=MAX_STEPS,
        gradient_checkpointing=True,
        fp16=(device == "cuda"),
        eval_strategy="steps",
        per_device_eval_batch_size=4,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS,
        logging_steps=LOG_STEPS,
        report_to=["none"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )

    print(f"\n[TRAIN] Starting... Train={len(train_data)} | Eval={len(eval_data)}")
    print(f"  Batch={BATCH_SIZE} | GradAccum={GRAD_ACCUM} | FP16={device=='cuda'}\n")

    trainer.train()

    print("[SAVE] Saving fine-tuned model...")
    model.save_pretrained(str(OUTPUT_DIR))
    processor.save_pretrained(str(OUTPUT_DIR))

    print(f"\n[DONE] Model saved to {OUTPUT_DIR}")
    print("  Update main.py: whisper.load_model() -> load from this path")


if __name__ == "__main__":
    finetune()
