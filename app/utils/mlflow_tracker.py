import mlflow
import pandas as pd
from pathlib import Path

class RAGExperimentTracker:
    def __init__(self, experiment_name="RAG_Optimization"):
        # Set the location where experiments are saved locally
        mlflow.set_tracking_uri("file:./mlflow_runs")
        mlflow.set_experiment(experiment_name)

    def start_run(self, run_name):
        """Context manager for an MLflow run."""
        return mlflow.start_run(run_name=run_name)

    def log_params(self, config_dict):
        """Logs all RAG hyperparameters."""
        mlflow.log_params(config_dict)

    def log_metrics_from_csv(self, csv_path, metric_prefix=""):
        """
        Reads the evaluation CSV, calculates averages, and logs to MLflow.
        """
        df = pd.read_csv(csv_path)
        
        # Determine metrics based on available columns
        prefix = f"{metric_prefix}_" if metric_prefix else ""
        metrics = {
            f"{prefix}avg_retrieval_quality": df["score_retrieval_quality"].mean(),
            f"{prefix}avg_faithfulness": df["score_llm_faithfulness"].mean()
        }
        
        # Add the specific hop-type metric
        if "score_reasoning_quality" in df.columns:
            metrics[f"{prefix}avg_reasoning_quality"] = df["score_reasoning_quality"].mean()
        elif "score_answer_relevance" in df.columns:
            metrics[f"{prefix}avg_answer_relevance"] = df["score_answer_relevance"].mean()
            
        mlflow.log_metrics(metrics)
        
        # Attach the actual CSV file as an 'Artifact' for future proof
        mlflow.log_artifact(csv_path)
        
        return metrics

    def set_run_tags(self, tags):
        """Adds tags like 'dataset_type': 'multihop'."""
        mlflow.set_tags(tags)
