# UCLAtron

### STILL DEVELOPING ###
Basically NYUtron and Gatortron except for UCLA

```
project_root/
│
├── data/
│    ├── train_data.csv
│    ├── dev_data.csv
│    ├── test_data.csv
│    └── ...
│
├── models/
│    ├── teacher_model/   # Directory for teacher weights (if fine-tuned)
│    ├── student_model/   # Directory for student weights (after distillation)
│    └── ...
│
├── main.py               # High-level script to run training/distillation
├── distillation.py       # Core logic for teacher -> student distillation
├── negative_distillation.py  # Implementation of “unlearning” strategies
├── evaluate_forgetting.py    # Evaluations to confirm knowledge removal
├── utils.py              # Common utility functions
└── requirements.txt

```

How to run standard distillation

```
python main.py \
  --teacher_model path/to/teacher \
  --student_model bert-base-uncased \
  --do_standard_distill \
  --train_file data/train_data.csv \
  --dev_file data/dev_data.csv \
  --test_file data/test_data.csv
```

how to run negative distillation

```
python main.py \
  --teacher_model path/to/teacher \
  --student_model bert-base-uncased \
  --do_negative_distill \
  --negative_concept "diabetes" \
  --train_file data/train_data.csv \
  --dev_file data/dev_data.csv \
  --test_file data/test_data.csv
```

### Notes
Notes and Next Steps
- Better “Negative” Distillation Criteria: The simplistic approach in negative_distillation.py penalizes matching teacher logits on concept-related examples. In practice, we might design more nuanced objectives (e.g., mask out certain logits, remove partial attention to concept tokens, or freeze certain layers).
- Probe Training: To truly verify forgetting, we’d build a small supervised probe to detect if the “negative concept” is still encoded in the latent representations.
- Membership Inference: Implementing a real membership inference attack requires creating an “attack model” to guess if a sample was in the training set.
- Progressive Forgetting: If we need multi-step forgetting (e.g., Teacher1 -> Student1 -> Teacher2 -> Student2, etc.), we can adapt the skeleton to chain these steps and keep track of intermediate models.
