import os
import json
import pandas as pd
import time
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

class RAGEvaluator:
    def __init__(self, model_name="gemini-2.5-flash"):
        """
        Initializes the evaluation agent using a high-reasoning model as a judge.
        """
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0
        )
        
    def _get_eval_prompt(self, is_multi_hop=False):
        """
        Returns the appropriate prompt template based on the complexity of the question.
        Literal JSON braces are escaped using double curly brackets.
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
            "metric_score": integer,
            "error_type": "string (None, Retrieval Gap, Reasoning Failure, Hallucination, or Incomplete)"
        }}}}
        """
        return PromptTemplate.from_template(template)

    def evaluate_file(self, file_path):
        """
        Processes a CSV file and fills in the evaluation scores.
        """
        path = Path(file_path)
        if not path.exists():
            print(f"File not found: {file_path}")
            return

        df = pd.read_csv(file_path)
        if "error_type" in df.columns:
            df["error_type"] = df["error_type"].astype("string")
        
        # Determine if this is a Multi-Hop or Single-Hop file based on columns
        is_multi_hop = "score_reasoning_quality" in df.columns
        metric_col = "score_reasoning_quality" if is_multi_hop else "score_answer_relevance"
        
        prompt_template = self._get_eval_prompt(is_multi_hop)
        chain = prompt_template | self.llm

        print(f"Starting automated evaluation for: {file_path}")
        
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
                response = chain.invoke({
                    "question": question,
                    "ground_truth": ground_truth,
                    "bot_answer": bot_answer
                })
                
                # Clean and parse JSON
                content = response.content.strip()
                if content.startswith("```json"):
                    content = content[7:-3].strip()
                elif content.startswith("```"):
                    content = content[3:-3].strip()
                
                result = json.loads(content)
                retrieval_score = result.get("retrieval_score")
                faithfulness_score = result.get("faithfulness_score")
                metric_score = result.get("metric_score")
                error_type = result.get("error_type")

                if metric_score is None:
                    raise KeyError("metric_score")
                
                # Update DataFrame
                df.at[idx, "score_retrieval_quality"] = retrieval_score
                df.at[idx, "score_llm_faithfulness"] = faithfulness_score
                df.at[idx, metric_col] = metric_score
                df.at[idx, "error_type"] = error_type
                
                print(f"Evaluated row {idx + 1}/{len(df)}")
                
                # Rate limiting prevention for Gemini API
                time.sleep(1) 
                
            except Exception as e:
                print(f"Error evaluating row {idx}: {str(e)}")

        # Save result
        output_name = path.stem + "_automated_eval.csv"
        output_path = path.parent / output_name
        df.to_csv(output_path, index=False)
        print(f"Evaluation complete. Saved to: {output_path}")

if __name__ == "__main__":
    evaluator = RAGEvaluator()

    evaluator.evaluate_file("./results/quality_baseline_scores.csv")
    evaluator.evaluate_file("./results/multihop_baseline_scores.csv")
