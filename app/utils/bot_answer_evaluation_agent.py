import os
import sys
import json
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config.experiment_config import RAGConfig

# Load environment variables
load_dotenv(REPO_ROOT / ".env")

class RAGEvaluator:
    def __init__(self, config: RAGConfig, retries: int = 1, sleep_seconds: float = 1.0):
        """
        Initializes the evaluation agent using a high-reasoning model as a judge.
        """
        self.llm = ChatGoogleGenerativeAI(
            model=config.eval_judge_model,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0
        )
        self.retries = retries
        self.sleep_seconds = sleep_seconds
        
    def _parse_response(self, response):
        content = response.content.strip()
        if content.startswith("```json"):
            content = content[7:-3].strip()
        elif content.startswith("```"):
            content = content[3:-3].strip()
        return json.loads(content)

    def _log_eval_error(self, path: Path, idx: int, content: str, error: str):
        log_path = path.parent / "eval_errors.log"
        with open(log_path, "a") as handle:
            handle.write(f"[row={idx}] {error}\n")
            handle.write(content)
            handle.write("\n---\n")

    def _get_eval_prompt(self, is_multi_hop=False):
        """
        Returns the appropriate prompt template based on the complexity of the question.
        """
        metric_name = "score_reasoning_quality" if is_multi_hop else "score_answer_relevance"
        metric_desc = (
            "score_reasoning_quality (1-5): Evaluate the logical connection between multiple retrieved facts." 
            if is_multi_hop else 
            "score_answer_relevance (1-5): Evaluate how directly and accurately the bot answered the specific question."
        )

        template = f"""
        You are a Senior Quality Assurance Auditor for an Enterprise RAG system. 
        Your task is to evaluate the bot's response against a provided Ground Truth.

        EVALUATION RUBRIC:
        1. score_retrieval_quality (1-5): Did the bot's answer contain the essential information found in the ground truth?
        2. score_llm_faithfulness (1-5): Did the bot avoid hallucinations? (5 = strictly grounded, 1 = made up external information).
        3. {metric_desc}

        INPUT DATA:
        - Question: {{question}}
        - Ground Truth: {{ground_truth}}
        - Bot Answer: {{bot_answer}}

        OUTPUT REQUIREMENTS:
        Return ONLY a JSON object with the following structure:
        {{{{
            "retrieval_score": integer,
            "faithfulness_score": integer,
            "{metric_name}": integer,
            "error_type": "string (None, Retrieval Gap, Reasoning Failure, Hallucination, or Incomplete)"
        }}}}
        """
        return PromptTemplate.from_template(template)

    def evaluate_file(self, file_path):
        """
        Processes a CSV file and fills in the evaluation scores.
        """
        path = Path(file_path)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.exists():
            print(f"File not found: {file_path}")
            return

        df = pd.read_csv(path)
        if "error_type" in df.columns:
            df["error_type"] = df["error_type"].astype("string")
        
        # Determine if this is a Multi-Hop or Single-Hop file based on columns
        is_multi_hop = "score_reasoning_quality" in df.columns
        metric_col = "score_reasoning_quality" if is_multi_hop else "score_answer_relevance"
        
        prompt_template = self._get_eval_prompt(is_multi_hop)
        chain = prompt_template | self.llm

        print(f"Starting automated evaluation for: {path}")
        
        for idx, row in df.iterrows():
            # Skip rows already processed
            if pd.notna(row.get("score_retrieval_quality")) and row.get("score_retrieval_quality") != "":
                continue

            # Standardize column mapping
            question = row.get("question") or row.get("user_input")
            ground_truth = row.get("ground_truth") or row.get("reference")
            bot_answer = row.get("bot_answer")

            if not question or not ground_truth or not bot_answer:
                continue

            try:
                result = None
                last_content = ""
                for attempt in range(self.retries + 1):
                    response = chain.invoke({
                        "question": question,
                        "ground_truth": ground_truth,
                        "bot_answer": bot_answer
                    })
                    last_content = response.content.strip()
                    result = self._parse_response(response)
                    retrieval_score = result.get("retrieval_score")
                    faithfulness_score = result.get("faithfulness_score")
                    metric_score = result.get(metric_col)
                    error_type = result.get("error_type")

                    if metric_score is not None:
                        break

                if metric_score is None:
                    raise KeyError(metric_col)

                # Update DataFrame
                df.at[idx, "score_retrieval_quality"] = retrieval_score
                df.at[idx, "score_llm_faithfulness"] = faithfulness_score
                df.at[idx, metric_col] = metric_score
                df.at[idx, "error_type"] = error_type
                
                print(f"Evaluated row {idx + 1}/{len(df)}")
                
                # Rate limiting prevention for Gemini API
                time.sleep(self.sleep_seconds)
                
            except Exception as e:
                self._log_eval_error(path, idx, last_content, str(e))
                print(f"Error evaluating row {idx}: {str(e)}")

        # Save result
        output_name = path.stem + "_automated_eval.csv"
        output_path = path.parent / output_name
        df.to_csv(output_path, index=False)
        print(f"Evaluation complete. Saved to: {output_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="*", default=None)
    parser.add_argument("--eval-judge-model", default=RAGConfig.eval_judge_model)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--sleep", type=float, default=1.0)
    args = parser.parse_args()

    cfg = RAGConfig(eval_judge_model=args.eval_judge_model)
    evaluator = RAGEvaluator(cfg, retries=args.retries, sleep_seconds=args.sleep)

    if args.files:
        for file_path in args.files:
            evaluator.evaluate_file(file_path)
    else:
        evaluator.evaluate_file(REPO_ROOT / "results" / "quality_baseline_scores.csv")
        evaluator.evaluate_file(REPO_ROOT / "results" / "multihop_baseline_scores.csv")
