from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

local_path = r"C:\Users\HOSEIN\PycharmProjects\Sentiment-Analysis-AP\app\models\bert_base_uncased"
model = AutoModelForSequenceClassification.from_pretrained(local_path,
                                                           num_labels=5,
                                                           local_files_only=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True)
