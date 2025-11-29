from transformers import AutoModelForSequenceClassification, AutoTokenizer
# import torch

# local_path = r"C:\Users\HOSEIN\PycharmProjects\Sentiment-Analysis-AP\app\models\bert_base_uncased"
# model = AutoModelForSequenceClassification.from_pretrained(local_path,
#                                                            num_labels=5,
#                                                            local_files_only=True)

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"

model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                            num_labels=5,
                                                           torch_dtype="auto",
                                                           device_map="auto")

tokenizer = AutoTokenizer.from_pretrained(model_name)
