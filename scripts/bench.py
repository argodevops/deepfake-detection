#!/usr/bin/env python3
import os
import glob
import json
import requests
from collections import Counter

API = os.getenv("API_URL", "http://localhost/predict")
REAL_DIR = os.getenv("REAL_DIR", "testdata/real")
FAKE_DIR = os.getenv("FAKE_DIR", "testdata/fake")
TIMEOUT = float(os.getenv("TIMEOUT", "120"))

def classify(path: str):
    with open(path, "rb") as f:
        r = requests.post(API, files={"file": f}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()

def run_dir(label: str, folder: str):
    paths = sorted(glob.glob(os.path.join(folder, "*")))
    results = []
    for p in paths:
        try:
            out = classify(p)
            results.append((p, label, out.get("result"), out.get("confidence") or out.get("score")))
        except Exception as e:
            results.append((p, label, "ERROR", str(e)))
    return results

def main():
    all_results = []
    all_results += run_dir("REAL", REAL_DIR)
    all_results += run_dir("FAKE", FAKE_DIR)

    counts = Counter()
    for _, truth, pred, _ in all_results:
        counts[(truth, pred)] += 1

    total = len(all_results)
    correct = sum(1 for _, t, p, _ in all_results if t == p)
    errors = sum(1 for _, _, p, _ in all_results if p == "ERROR")

    print("==== Confusion (truth, pred) ====")
    for k in sorted(counts.keys()):
        print(f"{k}: {counts[k]}")

    print("\n==== Summary ====")
    print("Total:", total)
    print("Correct:", correct)
    print("Errors:", errors)
    if total - errors > 0:
        print("Accuracy (excluding errors):", correct / (total - errors))

    print("\n==== Misclassifications ====")
    for p, truth, pred, score in all_results:
        if pred not in ("ERROR", truth):
            print(f"{truth} -> {pred}  score={score}  file={p}")

if __name__ == "__main__":
    main()

