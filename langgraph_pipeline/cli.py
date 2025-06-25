# langgraph_pipeline/cli.py

import logging
from datetime import datetime
from .dag import run_pipeline

# Setup structured logging
logging.basicConfig(
    filename="logs/pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def log_interaction(log_data):
    logging.info(f"Input: {log_data['text']}")
    logging.info(f"Predicted: {log_data['predicted_label']} | Confidence: {log_data['confidence']*100:.2f}%")
    
    if log_data["status"] == "fallback_needed":
        if log_data["corrected_by_user"]:
            logging.info("Fallback triggered -> User corrected label")
        else:
            logging.info("Fallback triggered -> User accepted prediction")
    
    logging.info(f"Final Label: {log_data['final_label']}")
    logging.info("-" * 60)

def main():
    print("=== Self-Healing Text Classifier (ATG DAG Project) ===\n")
    print("Type 'exit' to quit the program.\n")

    while True:
        user_input = input("Enter text: ").strip()
        if user_input.lower() == "exit":
            print("Exiting... Goodbye!")
            break

        log_data = run_pipeline(user_input)
        log_interaction(log_data)
        print("-" * 50 + "\n")

if __name__ == "__main__":
    main()