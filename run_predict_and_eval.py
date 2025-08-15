#!/usr/bin/env python3
# coding=utf-8
"""
Run prediction + evaluation multiple times, parse scores, and report averages.
All prediction is done in-process using a shared async chat client per run.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import asyncio
import subprocess
import evaluation_lib
from chat_completion_util import create_async_chat_client
import sys
from pathlib import Path
from statistics import mean
from datetime import datetime
import logging

import nltk
try:
    nltk.download('punkt_tab', quiet=True)
except Exception:
    # Best-effort; continue even if the optional resource isn't available
    pass
    
async def predict(
    inputs: list,
    client,
    model: str,
    max_concurrent: int,
    temperature: float | None,
    max_tokens: int | None,
    reasoning_effort: str | None,
) -> list[str]:
    """
    Generate responses for a list of prompt objects using the provided async client.
    """
    semaphore = asyncio.Semaphore(max_concurrent or 10)
    async def run_one(idx: int, prompt: str):
        async with semaphore:
            text = await client.chat_completion_with_retry(
                prompt_text=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
            )
            return idx, text

    tasks = [run_one(i, inp.prompt) for i, inp in enumerate(inputs)]
    results = await asyncio.gather(*tasks)
    results.sort(key=lambda x: x[0])
    return [r[1] for r in results]

def build_base_output_dir(model: str, tag: str | None = None, reasoning_effort: str | None = None) -> Path:
    base = model
    # Prefer explicit tag; else mirror run.sh behavior by suffixing reasoning_effort when provided
    if tag:
        base = f"{model}_{tag}"
    elif reasoning_effort:
        base = f"{model}_{reasoning_effort}"
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("results") / f"{base}_{ts}"


async def run_predict(
    input_data: Path,
    output_file: Path,
    model: str,
    *,
    client,
    max_concurrent: int,
    temperature: float | None,
    max_tokens: int | None,
    reasoning_effort: str | None,
) -> None:
    """Generate responses using the shared async client and write to output_file."""
    inputs = evaluation_lib.read_prompt_list(input_data)
    responses = await predict(
        inputs=inputs,
        client=client,
        model=model,
        max_concurrent=max_concurrent,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as fout:
        for p_idx, (inp, resp) in enumerate(zip(inputs, responses)):
            fout.write(json.dumps({
                "prompt": inp.prompt,
                "response": resp,
                "prompt_index": p_idx,
            }) + "\n")


def run_eval(
    input_data: Path,
    input_response_data: Path,
    output_dir: Path,
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "run_eval.py",
           "--input_data", str(input_data),
           "--input_response_data", str(input_response_data),
           "--output_dir", str(output_dir)]
    proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
    # Persist full log
    log_path = output_dir / "run_eval.log"
    log_path.write_text((proc.stdout or "") + (proc.stderr or ""))
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return (proc.stdout or "") + (proc.stderr or "")


def extract_scores(log_text: str, kind: str) -> tuple[float, float]:
    """
    Extract (prompt-level, instruction-level) for kind in {"strict","loose"}.
    Parses blocks like:
      .../eval_results_strict.jsonl Accuracy Scores:
      prompt-level: 0.6258503401360545
      instruction-level: 0.6567164179104478
    """
    marker = f"eval_results_{kind}.jsonl Accuracy Scores:"
    idx = log_text.rfind(marker)
    if idx == -1:
        raise RuntimeError(f"Could not find accuracy block for {kind}")
    block = log_text[idx: idx + 2000]
    m1 = re.search(r"prompt-level:\s*([0-9]*\.?[0-9]+)", block)
    m2 = re.search(r"instruction-level:\s*([0-9]*\.?[0-9]+)", block)
    if not (m1 and m2):
        raise RuntimeError(f"Could not parse metrics for {kind}")
    return float(m1.group(1)), float(m2.group(1))


async def main():
    ap = argparse.ArgumentParser(description="Repeat predict+eval and average scores")
    ap.add_argument("--model", default="gpt-oss-120b")
    ap.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY", None))
    ap.add_argument("--base_url", default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--max_tokens", type=int, default=None)
    ap.add_argument("--max_concurrent", type=int, default=10)
    ap.add_argument("--reasoning_effort", choices=["low", "medium", "high"], default=None)
    ap.add_argument("--input_data", default=str(Path("data") / "IFBench_test.jsonl"))
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--tag", default=None, help="Optional tag to suffix result dir (e.g., 'high')")
    ap.add_argument("--clean_base_dir", action="store_true", help="Remove base output dir before running")
    args = ap.parse_args()

    input_data = Path(args.input_data).resolve()
    base_dir = build_base_output_dir(args.model, args.tag, args.reasoning_effort)

    # Configure logging (console + file in base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("runner")
    logger.setLevel(logging.INFO)
    # Clear existing handlers to avoid duplicate logs if run twice in-process
    logger.handlers.clear()
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    fh = logging.FileHandler(base_dir / "orchestrator.log")
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    sh.setFormatter(fmt)
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)

    if args.clean_base_dir and base_dir.exists():
        logger.info("Cleaning base output dir: %s", base_dir)
        shutil.rmtree(base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

    strict_prompt_scores: list[float] = []
    strict_instr_scores: list[float] = []
    loose_prompt_scores: list[float] = []
    loose_instr_scores: list[float] = []

    # create a single async client for all repeats
    client = create_async_chat_client(api_key=args.api_key, base_url=args.base_url)
    for r in range(args.repeats):
        run_dir = base_dir / f"r{r:02d}"
        output_file = run_dir / f"{args.model}_responses.jsonl"

        logger.info("[Repeat %d/%d] Predicting -> %s", r + 1, args.repeats, output_file)
        await run_predict(
            input_data=input_data,
            output_file=output_file,
            model=args.model,
            client=client,
            max_concurrent=args.max_concurrent,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            reasoning_effort=args.reasoning_effort,
        )

        logger.info("[Repeat %d/%d] Evaluating -> %s", r + 1, args.repeats, run_dir)
        log_text = run_eval(
            input_data=input_data,
            input_response_data=output_file,
            output_dir=run_dir,
        )

        sp, si = extract_scores(log_text, "strict")
        lp, li = extract_scores(log_text, "loose")
        strict_prompt_scores.append(sp)
        strict_instr_scores.append(si)
        loose_prompt_scores.append(lp)
        loose_instr_scores.append(li)

    # Cleanup client
    await client.close()

    summary = {
        "repeats": args.repeats,
        "model": args.model,
        "averages": {
            "strict": {
                "prompt_level": mean(strict_prompt_scores),
                "instruction_level": mean(strict_instr_scores),
            },
            "loose": {
                "prompt_level": mean(loose_prompt_scores),
                "instruction_level": mean(loose_instr_scores),
            },
        },
        "per_repeat": {
            "strict": [
                {"prompt_level": p, "instruction_level": i}
                for p, i in zip(strict_prompt_scores, strict_instr_scores)
            ],
            "loose": [
                {"prompt_level": p, "instruction_level": i}
                for p, i in zip(loose_prompt_scores, loose_instr_scores)
            ],
        },
    }

    base_dir.mkdir(parents=True, exist_ok=True)
    summary_path = base_dir / "aggregate_scores.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    logger.info("================ Aggregated Scores ================")
    logger.info("%s", json.dumps(summary["averages"], indent=2))
    logger.info("Wrote aggregate summary to %s", summary_path)


if __name__ == "__main__":
    asyncio.run(main())
