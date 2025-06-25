# Self-Healing Text Classification DAG (ATG Technical Assignment)

## 🚀 Project Overview

This project implements a **self-healing text classification pipeline** using a fine-tuned **DistilBERT** model and a **LangGraph DAG**. The pipeline uses prediction confidence to determine whether to accept a classification, request user clarification, or escalate with fallback logic. This supports reliable and human-in-the-loop NLP workflows.

## 📁 Directory Structure

```
atg_self_healing_dag/
├── model/
│   ├── trainer.py          # Model fine-tuning script
│   └── model_files/        # Trained model files
├── langgraph_pipeline/
│   ├── cli.py              # Command-line interface
│   ├── dag.py              # LangGraph DAG runner
│   └── nodes.py            # DAG node logic (Inference, Confidence, Fallback)
├── logs/
│   └── pipeline.log        # Structured log output
├── requirements.txt        # All Python dependencies
└── README.md               # Project documentation
```

## 📦 Requirements

Create a virtual environment and install dependencies:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🏋️‍♂️ Model Training

Fine-tune the DistilBERT model on the HuggingFace Emotion dataset:

```bash
python model/trainer.py
```

- Dataset used: `emotion`
- Labels: `['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']`
- Model saved to: `model/model_files/`

---

## 🧠 LangGraph DAG Components

- **InferenceNode**: Runs the trained model on user input
- **ConfidenceCheckNode**: Accepts prediction if confidence >= 0.75
- **FallbackNode**: Asks user to confirm/correct label if confidence is too low

---

## 🖥️ Run the CLI

Launch the classification DAG from terminal:

```bash
python langgraph_pipeline/cli.py
```

Example flow:

```
Enter text: The movie was painfully slow and boring.
[InferenceNode] Predicted: sadness | Confidence: 74.25%
[ConfidenceCheckNode] Confidence too low ❌ — triggering fallback...
[FallbackNode] Could you clarify your intent?
0. sadness
1. joy
2. love
3. anger
4. fear
5. surprise
Enter the correct label number (0-5), or press Enter to accept prediction:
```

---

## 📝 Logs

All activity is logged in:

```
logs/pipeline.log
```
Contains:

- Prediction input, label, confidence
- Fallback decision & user corrections
- Final label

---

## 🔗 Download Full Project (with Model)

Due to size limitations on GitHub, the full project with the fine-tuned model is available here:

📦 [Download from Google Drive](https://drive.google.com/drive/folders/1nAtGogrqZcw5F3VJ8UqmY-AQ2aMgpVoq?usp=sharing)

## 👤 Author

Trinadh Katlagunta
B-Tech: CSE(AI & ML)
R.V.R & J.C College of Engineering

---

If you have any questions or issues running the project, feel free to raise an issue or contact me!
