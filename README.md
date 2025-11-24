# Sentiment Analysis API (BERT + FastAPI)

This project provides a production-ready **Sentiment Analysis REST API** built with **FastAPI** and powered by a locally loaded **BERT model**. The API processes input text and returns sentiment classification along with model confidence and a 1–5 star rating representation.

---

## 🚀 Features

* Implemented with **FastAPI**
* Utilizes **HuggingFace Transformers**
* Loads the model entirely from **local files** (no internet required)
* Automatically uses **GPU if available**
* Returns:

  * Sentiment label (`positive`, `neutral`, `negative`)
  * Confidence score
  * Star rating (1–5)

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the API

Start the FastAPI server with:

```bash
uvicorn app.main:app --reload
```

Once running:

* Base endpoint:
  `http://localhost:8000`

* Interactive Swagger docs:
  `http://localhost:8000/docs`

---

## 📮 Example Request

**POST** `/sentiment`

### Request body

```json
{
  "text": "The movie was good and enjoyable"
}
```

### Response

```json
{
  "label": "positive",
  "score": 0.9873,
  "stars": 5
}
```

---

## 🧠 How It Works

1. The input text is tokenized using the appropriate BERT tokenizer.
2. The locally stored model produces logits for each possible sentiment.
3. Softmax is applied to obtain probabilities.
4. The class with the highest probability is returned as the final prediction.
5. The model interprets predicted class IDs as:

   * 1 → very negative
   * 2 → negative
   * 3 → neutral
   * 4 → positive
   * 5 → very positive

---

## ⚠️ Input Constraints

* Maximum input size is **512 tokens**
  If exceeded, the API returns:

```json
{
  "detail": "Text too long"
}
```

---
## Local model setup

This project expects a local pretrained model directory at:

`app/models/bert_base_uncased/`

Place the following files inside that folder:
- config.json
- pytorch_model.bin
- vocab.txt
- tokenizer_config.json

### Option A — Copy from local machine
If you already have the model files on your machine, copy them into:
`app/models/bert_base_uncased/`

### Option B — Download from URL
If you host the model somewhere (Google Drive / Dropbox / private server), download and unzip into that folder. Example (Linux):

```bash
mkdir -p app/models/bert_base_uncased
wget -O model.zip "https://example.com/your-model.zip"
unzip model.zip -d app/models/bert_base_uncased
```


## 🙋 Author

**Hossein Khosravi**

Feel free to open issues, suggestions, or improvements.
