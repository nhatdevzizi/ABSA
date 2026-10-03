from datasets import load_dataset

ds = load_dataset("uitnlp/vietnamese_students_feedback")
train = load_dataset("uitnlp/vietnamese_students_feedback", split="train")
validation = load_dataset("uitnlp/vietnamese_students_feedback", split = "validation")

