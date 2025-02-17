"""
### STILL IMPLEMENTING ###
Implements a "negative distillation" approach to remove or forget specific concepts.
For example, we can penalize the student for matching the teacher's logits on
the 'negative_concept'.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from utils import load_dataset, create_collate_fn

def negative_distillation(
    teacher_model,
    student_model,
    teacher_tokenizer,
    student_tokenizer,
    train_file,
    dev_file,
    epochs,
    batch_size,
    device,
    negative_concept="diabetes"
):
    """
    Example approach:
    1. Identify samples in the training data related to the concept we want to forget.
    2. Instead of minimizing KL-divergence, we maximize it for that concept
       (or apply some penalty).
    3. For other samples, do normal distillation.
    """
    alpha = 0.5
    temperature = 2
    penalty_weight = 1.0  # Weight for penalty when concept is present

    # Load data
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

    teacher_model.eval()
    student_model.train()

    optimizer = torch.optim.AdamW(student_model.parameters(), lr=5e-5)
    kl_loss_fn = nn.KLDivLoss(reduction="batchmean")

    for epoch in range(epochs):
        for batch in train_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            texts = batch["texts"]  # raw text if you loaded it in the dataset

            with torch.no_grad():
                teacher_outputs = teacher_model(input_ids=input_ids,
                                                attention_mask=attention_mask)
                teacher_logits = teacher_outputs.logits

            student_outputs = student_model(input_ids=input_ids,
                                            attention_mask=attention_mask)
            student_logits = student_outputs.logits

            T = temperature
            teacher_probs = nn.functional.softmax(teacher_logits / T, dim=-1)
            student_log_probs = nn.functional.log_softmax(student_logits / T, dim=-1)

            # Normal KD loss
            kd_loss = kl_loss_fn(student_log_probs, teacher_probs) * (alpha * T * T)

            # Hard label loss
            ce_loss = nn.functional.cross_entropy(student_logits, labels) * (1.0 - alpha)

            # Identify if the negative concept is present
            # (You can use actual label names or text matching, etc.)
            negative_mask = [1 if negative_concept.lower() in txt.lower() else 0 
                             for txt in texts]
            negative_mask = torch.tensor(negative_mask, dtype=torch.float).to(device)

            # "Unlearning" approach:
            # We want to penalize similarity to teacher distribution on these samples.
            # One naive approach is to maximize the KL divergence for negative samples
            # => i.e., negative samples should NOT match teacher’s distribution.
            neg_kd_loss = kl_loss_fn(student_log_probs, teacher_probs) * (T * T)
            # But we *add* this to the total loss with a sign that encourages
            # the student to deviate from the teacher.
            # In practice, you might do: -(neg_kd_loss) or invert it with some factor.
            unlearning_loss = penalty_weight * negative_mask.mean() * neg_kd_loss

            # Combine losses:
            # For negative samples: total_loss will encourage "forgetting".
            # For normal samples: just normal KD + CE.
            total_loss = kd_loss + ce_loss + unlearning_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        print(f"[Epoch {epoch+1}/{epochs}] Loss: {total_loss.item():.4f}")

    print("Negative distillation completed.")
