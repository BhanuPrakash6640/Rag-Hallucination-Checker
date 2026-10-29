"""Measure the checker on RAGTruth, a public dataset of LLM answers with human-labelled hallucinations.

Setup (once):
    git clone https://github.com/ParticleMedia/RAGTruth.git data/RAGTruth

Examples:
    # Your checker on 300 test answers (threshold tuned on 300 train answers)
    python evaluate.py --method checker --limit 300 --tune 300

    # Only the QA task, with a faster NLI model
    python evaluate.py --method checker --tasks QA --nli-model cross-encoder/nli-deberta-v3-xsmall

    # Baseline: LLM-as-judge (uses LLM_* settings from .env). Price is USD per 1M tokens.
    python evaluate.py --method judge --limit 300 --price-in 0.15 --price-out 0.60

The task is response-level detection: does this answer contain any hallucination?
"""

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from hallcheck.config import Settings
from hallcheck.metrics import auroc, best_threshold, binary_metrics, percentile

TASKS = ["QA", "Summary", "Data2txt"]


# ---------------------------------------------------------------- data
def flatten(obj, prefix: str = "") -> list[str]:
    """Turn the Data2txt JSON records into simple 'key: value.' lines the NLI model can read."""
    if isinstance(obj, dict):
        return [line for k, v in obj.items() for line in flatten(v, f"{prefix}{k.replace('_', ' ')} ")]
    if isinstance(obj, list):
        return [line for item in obj for line in flatten(item, prefix)]
    return [f"{prefix.strip()}: {obj}."] if prefix.strip() else [f"{obj}."]


def make_evidence(source: dict) -> tuple[str, str | None]:
    """Return (evidence text, question or None) for one RAGTruth source."""
    info = source["source_info"]
    if source["task_type"] == "QA":
        return info["passages"], info.get("question")
    if source["task_type"] == "Summary":
        return info, None
    return "\n".join(flatten(info)), None


def load_ragtruth(data_dir: str, split: str, tasks: list[str], limit: int | None, seed: int) -> list[dict]:
    folder = Path(data_dir)
    responses_path, sources_path = folder / "response.jsonl", folder / "source_info.jsonl"
    if not responses_path.exists():
        raise SystemExit(
            f"RAGTruth not found in {folder}. Run:\n"
            "  git clone https://github.com/ParticleMedia/RAGTruth.git data/RAGTruth"
        )
    sources = {}
    with open(sources_path, encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            sources[s["source_id"]] = s
    examples = []
    with open(responses_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            s = sources[r["source_id"]]
            if r["split"] != split or s["task_type"] not in tasks:
                continue
            evidence, question = make_evidence(s)
            examples.append({
                "id": r["id"], "task": s["task_type"], "model": r["model"], "response": r["response"],
                "evidence": evidence, "question": question,
                "label": len(r["labels"]) > 0,  # True = contains a hallucination
                "spans": [l["text"] for l in r["labels"]],
            })
    random.Random(seed).shuffle(examples)
    return examples[:limit] if limit else examples


# ---------------------------------------------------------------- runners
def run_checker(examples: list[dict], checker) -> list[dict]:
    rows = []
    for ex in tqdm(examples, desc="checker"):
        res = checker.check(ex["response"], [ex["evidence"]])
        rows.append({
            **{k: ex[k] for k in ("id", "task", "model", "label", "spans")},
            "pred": res.hallucinated, "risk": res.risk_score, "latency_ms": res.latency_ms,
            "flagged": [c.claim for c in res.checked if c.verdict != "supported"],
        })
    return rows


def run_judge(examples: list[dict], judge) -> list[dict]:
    rows = []
    for ex in tqdm(examples, desc="llm-judge"):
        try:
            out = judge.judge(ex["evidence"], ex["response"], ex["question"])
        except Exception as e:  # keep going if one API call fails
            out = {"hallucinated": False, "latency_ms": 0, "prompt_tokens": 0, "completion_tokens": 0,
                   "parse_error": True, "unsupported_claims": [], "error": str(e)}
        rows.append({
            **{k: ex[k] for k in ("id", "task", "model", "label", "spans")},
            "pred": out["hallucinated"], "risk": float(out["hallucinated"]), "latency_ms": out["latency_ms"],
            "prompt_tokens": out["prompt_tokens"], "completion_tokens": out["completion_tokens"],
            "parse_error": out.get("parse_error", False), "flagged": out.get("unsupported_claims", []),
        })
    return rows


# ---------------------------------------------------------------- reporting
def summarize(rows: list[dict], pred_key: str = "pred") -> dict:
    y = [r["label"] for r in rows]
    out = binary_metrics(y, [r[pred_key] for r in rows])
    out["auroc"] = auroc(y, [r["risk"] for r in rows])
    out["n"] = len(rows)
    out["hallucination_rate"] = sum(y) / max(1, len(y))
    return out


def fmt(m: dict) -> str:
    return (f"n={m['n']:<5} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} "
            f"Acc={m['accuracy']:.3f} AUROC={m['auroc']:.3f} (base rate {m['hallucination_rate']:.2f})")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data/RAGTruth/dataset")
    p.add_argument("--method", choices=["checker", "judge"], default="checker")
    p.add_argument("--split", default="test")
    p.add_argument("--tasks", nargs="+", default=TASKS, choices=TASKS)
    p.add_argument("--limit", type=int, default=300, help="number of test answers (0 = all 2700)")
    p.add_argument("--tune", type=int, default=0, help="train answers used to pick the best risk threshold")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--nli-model", help="override NLI_MODEL")
    p.add_argument("--price-in", type=float, default=0.0, help="judge: USD per 1M input tokens")
    p.add_argument("--price-out", type=float, default=0.0, help="judge: USD per 1M output tokens")
    p.add_argument("--out", default="results")
    args = p.parse_args()

    settings = Settings()
    if args.nli_model:
        settings.nli_model = args.nli_model
    test = load_ragtruth(args.data_dir, args.split, args.tasks, args.limit or None, args.seed)
    print(f"Loaded {len(test)} {args.split} answers ({', '.join(args.tasks)})")

    report: dict = {"method": args.method, "tasks": args.tasks, "split": args.split, "seed": args.seed}
    if args.method == "checker":
        from hallcheck.verifier import HallucinationChecker

        checker = HallucinationChecker.from_settings(settings)
        report["nli_model"] = settings.nli_model
        rows = run_checker(test, checker)
        if args.tune:
            train = load_ragtruth(args.data_dir, "train", args.tasks, args.tune, args.seed)
            train_rows = run_checker(train, checker)
            thr, train_f1 = best_threshold([r["label"] for r in train_rows], [r["risk"] for r in train_rows])
            for r in rows:
                r["pred_tuned"] = r["risk"] >= thr
            report["tuned_threshold"] = thr
            report["tuned_train_f1"] = train_f1
    else:
        from hallcheck.generator import Generator
        from hallcheck.judge import LLMJudge

        report["judge_model"] = settings.llm_model
        rows = run_judge(test, LLMJudge(Generator(settings)))
        tokens_in = sum(r["prompt_tokens"] for r in rows)
        tokens_out = sum(r["completion_tokens"] for r in rows)
        cost = tokens_in / 1e6 * args.price_in + tokens_out / 1e6 * args.price_out
        report["cost_usd_per_1000_answers"] = cost / max(1, len(rows)) * 1000
        report["parse_errors"] = sum(r["parse_error"] for r in rows)

    # overall + per task
    report["overall"] = summarize(rows)
    if "pred_tuned" in rows[0]:
        report["overall_tuned"] = summarize(rows, "pred_tuned")
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)
    report["per_task"] = {t: summarize(rs, "pred_tuned" if "pred_tuned" in rs[0] else "pred") for t, rs in by_task.items()}
    lat = [r["latency_ms"] for r in rows]
    report["latency_ms"] = {"mean": sum(lat) / len(lat), "p50": percentile(lat, 50), "p95": percentile(lat, 95)}

    # print
    print(f"\n=== {args.method} ({report.get('nli_model') or report.get('judge_model')}) ===")
    print("Overall (default rules):   " + fmt(report["overall"]))
    if "overall_tuned" in report:
        print(f"Overall (tuned thr={report['tuned_threshold']:.3f}): " + fmt(report["overall_tuned"]))
    for t, m in report["per_task"].items():
        print(f"  {t:<9}" + fmt(m))
    l = report["latency_ms"]
    print(f"Latency per answer: mean {l['mean']:.0f} ms, p50 {l['p50']:.0f} ms, p95 {l['p95']:.0f} ms")
    if "cost_usd_per_1000_answers" in report:
        print(f"Cost: ${report['cost_usd_per_1000_answers']:.3f} per 1,000 answers "
              f"({report['parse_errors']} unparseable replies)")

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (out / f"{args.method}_{stamp}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with open(out / f"{args.method}_{stamp}_rows.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nSaved report and per-answer results to {out}/")


if __name__ == "__main__":
    main()
