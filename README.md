# UCLAtron
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
