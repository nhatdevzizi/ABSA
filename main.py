from datasets import load_dataset
import torch


device = "cuda" if torch.cuda.is_available() else "cpu"

# ds = load_dataset("uitnlp/vietnamese_students_feedback")
# train = load_dataset("uitnlp/vietnamese_students_feedback", split="train", streaming = True)
# validation = load_dataset("uitnlp/vietnamese_students_feedback", split = "validation")

# first_1000 = train.take(1000)

def main():
    x = torch.rand(5,3)
    print(x)

if __name__ == "__main__":
    main()