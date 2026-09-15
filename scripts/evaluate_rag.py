import os
import sys
import json
import pandas as pd
from langchain_ollama import ChatOllama
from loguru import logger

# Allow running as a script from the repo root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.rag.engine import RAGEngine

TEST_SET_PATH = "tests/data/test_set.json"
REPORT_PATH = "logs/evaluation_report_v2.csv"

FAITHFULNESS_PROMPT = """You are an expert judge. Evaluate the FAITHFULNESS of the following answer based ONLY on the provided context.

CONTEXT:
{context}

ANSWER:
{answer}

Score from 0.0 to 1.0, where 1.0 means the answer is entirely supported by the context, and 0.0 means it is not supported at all.
Return ONLY a JSON object of the form {{"score": <number>}}."""

RELEVANCE_PROMPT = """You are an expert judge. Evaluate the RELEVANCE of the following answer to the given question.

QUESTION:
{question}

ANSWER:
{answer}

Score from 0.0 to 1.0, where 1.0 means the answer fully addresses the question, and 0.0 means it is irrelevant.
Return ONLY a JSON object of the form {{"score": <number>}}."""

def parse_score(text: str):
    """Reads the 'score' field from the judge's JSON reply. Returns None if it is missing or out of range."""
    try:
        score = float(json.loads(text)["score"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    if not 0.0 <= score <= 1.0:
        return None
    return score

def judge(judge_llm, prompt: str, label: str):
    """Asks the judge for a score. Failures return None so they are excluded from the means, not counted as 0."""
    try:
        response = judge_llm.invoke(prompt).content.strip()
    except Exception as e:
        logger.error(f"Judge call failed ({label}): {e}")
        return None
    logger.debug(f"Judge RAW ({label}): {response}")
    score = parse_score(response)
    if score is None:
        logger.warning(f"Unparseable judge output ({label}): {response!r}")
    return score

def run_custom_evaluation():
    engine = RAGEngine()
    judge_llm = ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3"), temperature=0, format="json")

    # 1. Load Test Set
    with open(TEST_SET_PATH, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    results = []
    logger.info(f"Starting Custom Evaluation for {len(test_data)} cases...")

    for item in test_data:
        question = item["question"]
        logger.info(f"Testing Question: {question}")

        # Run RAG
        rag_output = engine.query(question)
        answer = rag_output["answer"]
        context_str = "\n---\n".join(rag_output.get("contexts", []))

        # Judge
        faith_score = judge(judge_llm, FAITHFULNESS_PROMPT.format(context=context_str, answer=answer), "Faithfulness")
        rel_score = judge(judge_llm, RELEVANCE_PROMPT.format(question=question, answer=answer), "Relevance")

        results.append({
            "question": question,
            "answer": answer,
            "faithfulness": faith_score,
            "relevance": rel_score
        })
        logger.info(f"Scores -> Faithfulness: {faith_score}, Relevance: {rel_score}")

    # 2. Summary
    df = pd.DataFrame(results)
    print("\n--- PERFORMANCE REPORT ---")
    print(df[["question", "faithfulness", "relevance"]])
    print("\nMEAN SCORES (unparseable judge outputs excluded):")
    print(df[["faithfulness", "relevance"]].mean())
    print("\nUNPARSEABLE JUDGE OUTPUTS:")
    print(df[["faithfulness", "relevance"]].isna().sum())

    # Save
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    df.to_csv(REPORT_PATH, index=False)
    logger.success(f"Evaluation complete! Report saved to {REPORT_PATH}")

if __name__ == "__main__":
    run_custom_evaluation()
