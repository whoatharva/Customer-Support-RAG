"""Live end-to-end Ragas evaluation for the Customer Support RAG system.

Unlike a static scorer, this harness runs every question in the golden set
(`evaluation/test_cases.py`) through the REAL pipeline — the same
`handle_chat()` path the API uses — captures the generated answer and the
retrieved citation chunks, and scores them with Ragas against the reference
answers.

Metrics (all LLM/embedding judged):
  answer_relevancy    how well the answer addresses the question
  faithfulness        fraction of answer claims grounded in the retrieved context
  context_recall      how much of the ground truth the retrieved context covers
  context_precision   fraction of retrieved context that is relevant
  answer_correctness  answer vs. ground_truth (semantic + factual overlap)

The judge LLM + embeddings reuse the app's own Azure OpenAI configuration from
`app.config.settings` (single `.env`), so scoring uses the same models the
system runs on — no separate AZURE_OPENAI_* variables to set.

Run (requires .env with Azure + Qdrant + Supabase reachable and the KB ingested):
  uv run python evaluation/ragas_eval.py                 # full set
  uv run python evaluation/ragas_eval.py --limit 2       # smoke test
  uv run python evaluation/ragas_eval.py --categories shipping,payment
  uv run python evaluation/ragas_eval.py --out evaluation/results

Setup (already in pyproject.toml):
  uv add ragas datasets langchain-openai
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Make both the project root (for `app`) and this dir (for `test_cases`)
# importable regardless of the cwd the script is launched from.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dotenv import load_dotenv

load_dotenv()


# ── Judge LLM + embeddings (reuse the app's Azure config) ─────────────────────

def _build_judge():
    """Construct the Ragas judge LLM + embeddings from app settings.

    Kept in a function so `--help` and import-time errors don't require a live
    Azure connection.
    """
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

    from app.config import settings

    api_version = "2024-08-01-preview"

    llm = LangchainLLMWrapper(AzureChatOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        azure_deployment=settings.azure_openai_deployment_name,
        api_version=api_version,
        temperature=0.0,
    ))
    embeddings = LangchainEmbeddingsWrapper(AzureOpenAIEmbeddings(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        azure_deployment=settings.azure_openai_embedding_deployment,
        api_version=api_version,
    ))
    model_info = {
        "judge_llm": settings.azure_openai_deployment_name,
        "judge_embeddings": settings.azure_openai_embedding_deployment,
    }
    return llm, embeddings, model_info


# ── Live pipeline runner ──────────────────────────────────────────────────────

def run_pipeline(case: dict) -> tuple[str, list[str]]:
    """Run one question through the real RAG pipeline.

    Returns (answer, contexts). A fresh session_id per case keeps each turn
    history-free so `process_query` can't rewrite one question using another.
    """
    from app.services.chat_service import handle_chat
    from app.schemas import ChatRequest

    request = ChatRequest(session_id=f"eval_{case['id']}", query=case["question"])
    resp = handle_chat(request, user_email="")
    answer = resp.answer or ""
    contexts = [c.text for c in resp.citations if getattr(c, "text", "")]
    return answer, contexts


def _generate(cases: list[dict]) -> tuple[list[dict], list[dict]]:
    """Run all cases through the pipeline. Returns (rows, failures)."""
    rows, failures = [], []
    for i, case in enumerate(cases, 1):
        cid = case["id"]
        print(f"[{i}/{len(cases)}] {cid} … ", end="", flush=True)
        try:
            answer, contexts = run_pipeline(case)
        except Exception as e:  # keep going; one bad case shouldn't kill the run
            print(f"FAILED ({type(e).__name__}: {e})")
            failures.append({"id": cid, "error": f"{type(e).__name__}: {e}"})
            continue
        print(f"ok ({len(contexts)} ctx)")
        rows.append({
            "id": cid,
            "category": case.get("category", ""),
            "user_input": case["question"],
            "response": answer,
            "retrieved_contexts": contexts if contexts else [""],
            "reference": case["ground_truth"],
        })
    return rows, failures


# ── Reporting ─────────────────────────────────────────────────────────────────

_METRIC_COLS = [
    "answer_relevancy",
    "faithfulness",
    "context_recall",
    "context_precision",
    "answer_correctness",
]


def _means(df) -> dict:
    """Mean of each metric column present in `df` (NaN-safe)."""
    return {c: float(df[c].mean()) for c in _METRIC_COLS if c in df.columns}


def _write_results(df, model_info: dict, failures: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"ragas_{stamp}.csv"
    json_path = out_dir / f"ragas_{stamp}_summary.json"

    df.to_csv(csv_path, index=False)

    metric_cols = [c for c in _METRIC_COLS if c in df.columns]

    # Per-category means so out_of_scope (which SHOULD score ~0) doesn't silently
    # drag down the in-scope numbers when read as a single blended average.
    per_category = {
        str(cat): _means(grp)
        for cat, grp in df.groupby("category", sort=True)
    } if "category" in df.columns else {}

    # In-scope aggregate excludes the out_of_scope category.
    in_scope_df = df[df["category"] != "out_of_scope"] if "category" in df.columns else df

    # Full per-case scores — the breakdown that was previously only in the CSV.
    per_case = [
        {
            "id": row.get("id"),
            "category": row.get("category"),
            **{c: (None if _is_nan(row.get(c)) else float(row.get(c))) for c in metric_cols},
        }
        for row in df.to_dict("records")
    ]

    summary = {
        "timestamp": stamp,
        "models": model_info,
        "num_cases_scored": int(len(df)),
        "num_failures": len(failures),
        "failures": failures,
        "mean_scores_all": _means(df),
        "mean_scores_in_scope": _means(in_scope_df),
        "mean_scores_by_category": per_category,
        "per_case": per_case,
    }
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote:\n  {csv_path}\n  {json_path}")


def _is_nan(v) -> bool:
    try:
        return v != v  # NaN is the only value not equal to itself
    except Exception:
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Live Ragas evaluation of the RAG pipeline.")
    parser.add_argument("--limit", type=int, default=None, help="only run the first N cases (smoke test)")
    parser.add_argument("--categories", type=str, default=None,
                        help="comma-separated categories to include (e.g. shipping,payment)")
    parser.add_argument("--out", type=str, default="evaluation/results", help="output directory for results")
    args = parser.parse_args()

    from test_cases import TEST_CASES

    cases = TEST_CASES
    if args.categories:
        wanted = {c.strip() for c in args.categories.split(",") if c.strip()}
        cases = [c for c in cases if c.get("category") in wanted]
    if args.limit:
        cases = cases[: args.limit]

    if not cases:
        print("No test cases match the given filters.", file=sys.stderr)
        return 1

    # 1) Build the judge up front so a misconfiguration fails fast, before we
    #    spend time/tokens running the pipeline.
    try:
        llm, embeddings, model_info = _build_judge()
    except Exception as e:
        print(f"Failed to initialise Ragas judge (check .env / Azure settings): {e}", file=sys.stderr)
        return 2

    # 2) Generate answers + contexts live.
    print(f"Running {len(cases)} case(s) through the live RAG pipeline…\n")
    rows, failures = _generate(cases)
    if not rows:
        print("\nAll cases failed to generate — nothing to score.", file=sys.stderr)
        for f in failures:
            print(f"  - {f['id']}: {f['error']}", file=sys.stderr)
        return 3

    # 3) Score with Ragas.
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy, faithfulness, context_recall,
        context_precision, answer_correctness,
    )

    dataset = Dataset.from_list([
        {
            "user_input": r["user_input"],
            "response": r["response"],
            "retrieved_contexts": r["retrieved_contexts"],
            "reference": r["reference"],
        }
        for r in rows
    ])

    print(f"\nScoring {len(dataset)} case(s) with Ragas "
          f"(judge={model_info['judge_llm']})…")
    metrics = [answer_relevancy, faithfulness, context_recall,
               context_precision, answer_correctness]
    result = evaluate(dataset=dataset, metrics=metrics, llm=llm, embeddings=embeddings)

    df = result.to_pandas()
    # Attach id/category for readable output & CSV.
    df.insert(0, "id", [r["id"] for r in rows])
    df.insert(1, "category", [r["category"] for r in rows])

    print("\n=== Ragas Evaluation Results ===")
    show_cols = ["id", "category"] + [c for c in _METRIC_COLS if c in df.columns]
    with_pd_opts(lambda: print(df[show_cols].to_string(index=False)))

    print("\n=== Mean Scores (all cases) ===")
    for col in _METRIC_COLS:
        if col in df.columns:
            print(f"  {col:<20} {df[col].mean():.3f}")

    if "category" in df.columns and (df["category"] == "out_of_scope").any():
        in_scope = df[df["category"] != "out_of_scope"]
        print("\n=== Mean Scores (in-scope only, excludes out_of_scope) ===")
        for col in _METRIC_COLS:
            if col in df.columns:
                print(f"  {col:<20} {in_scope[col].mean():.3f}")

    if failures:
        print(f"\n{len(failures)} case(s) failed to generate and were excluded:")
        for f in failures:
            print(f"  - {f['id']}: {f['error']}")

    _write_results(df, model_info, failures, Path(args.out))
    return 0


def with_pd_opts(fn):
    import pandas as pd
    with pd.option_context("display.max_colwidth", 40, "display.width", 200):
        return fn()


if __name__ == "__main__":
    raise SystemExit(main())
