# langgraph_pipeline/dag.py

from .nodes import InferenceNode, ConfidenceCheckNode, FallbackNode

def run_pipeline(input_text):
    log_data = {}

    # Step 1: Inference Node
    print("[InferenceNode] Running model inference...")
    prediction_output = InferenceNode.run(input_text)
    print(f"[InferenceNode] Predicted: {prediction_output['predicted_label']} | Confidence: {prediction_output['confidence']*100:.2f}%")
    log_data.update(prediction_output)

    # Step 2: Confidence Check Node
    print("[ConfidenceCheckNode] Evaluating confidence...")
    prediction_output = ConfidenceCheckNode.run(prediction_output)

    if prediction_output["status"] == "accepted":
        print(f"[ConfidenceCheckNode] Prediction accepted ✅")
        prediction_output["final_label"] = prediction_output["predicted_label"]
        prediction_output["corrected_by_user"] = False
    else:
        print("[ConfidenceCheckNode] Confidence too low ❌ — triggering fallback...")
        prediction_output = FallbackNode.run(prediction_output)

    # Final Output
    print(f"[Final Decision] Label: {prediction_output['final_label']}")

    log_data.update({
        "final_label": prediction_output["final_label"],
        "corrected_by_user": prediction_output["corrected_by_user"],
        "status": prediction_output["status"]
    })

    return log_data
