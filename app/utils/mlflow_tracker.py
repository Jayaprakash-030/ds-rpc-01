from contextlib import nullcontext

try:
    import mlflow
except ModuleNotFoundError:
    mlflow = None
import pandas as pd
from pathlib import Path

class RAGExperimentTracker:
    def __init__(self, experiment_name="RAG_Optimization"):
        self.enabled = mlflow is not None
        if self.enabled:
            # Set the location where experiments are saved locally
            mlflow.set_tracking_uri("file:./mlflow_runs")
            mlflow.set_experiment(experiment_name)
        else:
            print("[mlflow_tracker] mlflow not installed; running without MLflow logging.")

    def start_run(self, run_name):
        """Context manager for an MLflow run."""
        if not self.enabled:
            return nullcontext()
        return mlflow.start_run(run_name=run_name)

    def log_params(self, config_dict):
        """Logs all RAG hyperparameters."""
        if not self.enabled:
            return
        mlflow.log_params(config_dict)

    def log_metrics_from_csv(self, csv_path, metric_prefix=""):
        """
        Reads the evaluation CSV, calculates averages, and logs to MLflow.
        """
        df = pd.read_csv(csv_path)
        
        # Determine metrics based on available columns (RAGAS-only; no bot evaluation agent)
        prefix = f"{metric_prefix}_" if metric_prefix else ""
        metrics = {}

        # RAGAS metrics
        for col in ("context_precision", "context_recall", "faithfulness", "answer_relevancy"):
            if col in df.columns:
                s = pd.to_numeric(df[col], errors="coerce").dropna()
                if not s.empty:
                    metrics[f"{prefix}avg_{col}"] = float(s.mean())

        # Retrieval + context metrics if available
        for col in ("precision", "recall", "retrieved_contexts_count"):
            if col in df.columns:
                s = pd.to_numeric(df[col], errors="coerce").dropna()
                if not s.empty:
                    metrics[f"{prefix}avg_{col}"] = float(s.mean())

        # Latency percentiles (ms) if available
        if "latency_ms" in df.columns:
            latency = pd.to_numeric(df["latency_ms"], errors="coerce").dropna()
            if not latency.empty:
                metrics[f"{prefix}latency_p50_ms"] = float(latency.quantile(0.50))
                metrics[f"{prefix}latency_p95_ms"] = float(latency.quantile(0.95))
                metrics[f"{prefix}latency_mean_ms"] = float(latency.mean())

        # Retrieval/generation timing if available
        for col in ("retrieval_ms", "generation_ms", "total_ms"):
            if col in df.columns:
                s = pd.to_numeric(df[col], errors="coerce").dropna()
                if not s.empty:
                    metrics[f"{prefix}{col}_mean"] = float(s.mean())

        if self.enabled:
            mlflow.log_metrics(metrics)
        
        # Attach the actual CSV file as an 'Artifact' for future proof
        if self.enabled:
            mlflow.log_artifact(csv_path)
        
        return metrics

    def set_run_tags(self, tags):
        """Adds tags like 'dataset_type': 'multihop'."""
        if not self.enabled:
            return
        mlflow.set_tags(tags)
