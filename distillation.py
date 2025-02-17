"""
### STILL IMPLEMENTING ###

Implements standard teacher -> student knowledge distillation.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from utils import load_dataset, create_collate_fn

def standard_distillation(
    teacher_model,
    student_model,
    teacher_tokenizer,
    student_tokenizer,
    train_file,
    dev_file,
    epochs,
    batch_size,
    device
):
    # We assume a classification task for simplicity.

    # Hyperparams
    alpha = 0.5      # Weight for KD loss
    temperature = 2  # Temperature for softmax distillation

    # Prepare Data
    train_dataset = load_dataset(train_file, teacher_tokenizer)
    dev_dataset = load_dataset(dev_file, teacher_tokenizer)

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=create_collate_fn(teacher_tokenizer)
    )
    dev_dataloader = DataLoader(
        dev_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=create_collate_fn(teacher_tokenizer)
    )

    # Set models in training/eval mode
    teacher_model.eval()
    student_model.train()

    optimizer = torch.optim.AdamW(student_model.parameters(), lr=5e-5)
    loss_fn = nn.KLDivLoss(reduction="batchmean")

    for epoch in range(epochs):
        for batch in train_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.no_grad():
                teacher_outputs = teacher_model(input_ids=input_ids,
                                                attention_mask=attention_mask)
                teacher_logits = teacher_outputs.logits

            student_outputs = student_model(input_ids=input_ids,
                                            attention_mask=attention_mask)
            student_logits = student_outputs.logits

            # Soft Targets (teacher logits -> softmax)
            T = temperature
            teacher_probs = nn.functional.softmax(teacher_logits / T, dim=-1)
            student_log_probs = nn.functional.log_softmax(student_logits / T, dim=-1)

            kd_loss = loss_fn(student_log_probs, teacher_probs) * (alpha * T * T)

            # Hard label loss
            ce_loss = nn.functional.cross_entropy(student_logits, labels) * (1.0 - alpha)

            total_loss = kd_loss + ce_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        print(f"[Epoch {epoch+1}/{epochs}] Loss: {total_loss.item():.4f}")

    # Optionally evaluate on dev set here...
    print("Standard distillation completed.")
