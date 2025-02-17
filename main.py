"""
High Level PSEUDOCODE
#### CURRENTLY DEVELOPING ####
High-level script to run:
1. Teacher model training/fine-tuning (optional).
2. Negative or selective distillation to student model.
3. Evaluation of forgetting (before vs. after).
"""
import argparse
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from distillation import standard_distillation
from negative_distillation import negative_distillation
from evaluate_forgetting import run_evaluation

def parse_args():
    parser = argparse.ArgumentParser(description="Unlearning via negative distillation.")
    parser.add_argument("--teacher_model", type=str, default="dmis-lab/biobert-base-cased-v1.1",
                        help="Path or name of the teacher model.")
    parser.add_argument("--student_model", type=str, default="bert-base-uncased",
                        help="Path or name of the student model.")
    parser.add_argument("--negative_concept", type=str, default="diabetes",
                        help="Concept/class label to unlearn.")
    parser.add_argument("--output_dir", type=str, default="models/student_model",
                        help="Where to save the distilled student model.")
    parser.add_argument("--train_file", type=str, default="data/train_data.csv",
                        help="Training dataset.")
    parser.add_argument("--dev_file", type=str, default="data/dev_data.csv",
                        help="Dev/validation dataset.")
    parser.add_argument("--test_file", type=str, default="data/test_data.csv",
                        help="Test dataset.")
    parser.add_argument("--do_standard_distill", action="store_true",
                        help="Perform standard knowledge distillation.")
    parser.add_argument("--do_negative_distill", action="store_true",
                        help="Perform negative distillation (unlearning).")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Teacher Model & Tokenizer
    teacher_model = AutoModelForSequenceClassification.from_pretrained(args.teacher_model).to(device)
    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)

    # Optionally, fine-tune teacher model here if needed. 
    # For brevity, we assume teacher is pre-trained/fine-tuned.

    # Load or Initialize Student Model
    student_model = AutoModelForSequenceClassification.from_pretrained(args.student_model).to(device)
    student_tokenizer = AutoTokenizer.from_pretrained(args.student_model)

    # Perform standard knowledge distillation (optional)
    if args.do_standard_distill:
        print("Running standard knowledge distillation...")
        standard_distillation(
            teacher_model,
            student_model,
            teacher_tokenizer,
            student_tokenizer,
            args.train_file,
            args.dev_file,
            args.epochs,
            args.batch_size,
            device
        )

    # Perform negative (unlearning) distillation
    if args.do_negative_distill:
        print(f"Running negative distillation to forget concept: {args.negative_concept}")
        negative_distillation(
            teacher_model,
            student_model,
            teacher_tokenizer,
            student_tokenizer,
            args.train_file,
            args.dev_file,
            args.epochs,
            args.batch_size,
            device,
            negative_concept=args.negative_concept
        )

    # Save the (potentially updated) student model
    student_model.save_pretrained(args.output_dir)
    student_tokenizer.save_pretrained(args.output_dir)

    # Evaluation / Verification of forgetting
    run_evaluation(
        student_model, 
        student_tokenizer,
        args.test_file,
        device,
        negative_concept=args.negative_concept
    )

if __name__ == "__main__":
    main()
