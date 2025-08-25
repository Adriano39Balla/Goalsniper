#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_predictions.py

Evaluate precision metrics for trained models and saved predictions.

Expected project layout (flexible):
  artifacts/
    <run-id>/
      preds_<split>.csv         # y_true, y_prob_[class]..., optional: y_pred
      model_meta.json           # optional metadata (labels, thresholds, etc.)
      precision_report_<split>.json
      precision_report_<split>.csv
      pr_curve_<split>.png

Typical usage:
  python analyze_predictions.py \
    --run-id 2025-08-25_12-00-00 \
    --artifacts-dir artifacts \
    --split val \
    --threshold 0.5 \
    --per-class \
    --plot

Notes:
- Works for binary or multi-class (one-vs-rest) probability outputs.
- If y_pred is absent, it will threshold the max-prob (multiclass) or positive prob (binary).
- Threshold can be a single float for binary, or "auto" to pick the best threshold by F1 on the split.
- For multi-class, thresholding is applied one-vs-rest per class for per-class precision; argmax is used for overall precision.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    classification_report,
    precision_recall_curve,
    average_precision_score,
)

# Optional plotting (guarded so headless envs are fine)
try:
    import matplotlib.pyplot as plt
    _HAS_PLT = True
except Exception:  # pragma: no cover
    _HAS_PLT = False


# -----------------------------
# I/O helpers
# -----------------------------

def _default_preds_path(artifacts_dir: Path, run_id: str, split: str) -> Path:
    return artifacts_dir / run_id / f"preds_{split}.csv"

def _default_meta_path(artifacts_dir: Path, run_id: str) -> Path:
    return artifacts_dir / run_id / "model_meta.json"

def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Core logic
# -----------------------------

def infer_label_columns(df: pd.DataFrame) -> Tuple[str, List[str], Optional[str]]:
    """
    Infer ground-truth label column, prob columns, and optional predicted label column.
    Heuristics:
      - y_true: one of ["y_true", "label", "target"]
      - y_pred: one of ["y_pred", "pred"]
      - prob columns: columns starting with "y_prob_" OR named "prob" (binary)
    """
    y_true_candidates = [c for c in ["y_true", "label", "target"] if c in df.columns]
    if not y_true_candidates:
        raise ValueError("Could not find ground-truth column (expected one of y_true/label/target).")
    y_true_col = y_true_candidates[0]

    y_pred_col = None
    for c in ["y_pred", "pred"]:
        if c in df.columns:
            y_pred_col = c
            break

    # probability columns
    prob_cols = [c for c in df.columns if c.startswith("y_prob_")]
    if not prob_cols and "prob" in df.columns:
        prob_cols = ["prob"]  # treat as positive-class prob for binary

    if not prob_cols and y_pred_col is None:
        raise ValueError("No probability columns found (y_prob_* or prob), and y_pred is missing. "
                         "Provide either probabilities or predicted labels.")

    return y_true_col, prob_cols, y_pred_col


def is_binary(prob_cols: List[str], df: pd.DataFrame, y_true_col: str) -> bool:
    if prob_cols == ["prob"]:
        return True
    if len(prob_cols) <= 2:
        n_unique = df[y_true_col].nunique(dropna=True)
        return n_unique <= 2
    return False


def get_classes(prob_cols: List[str], df: pd.DataFrame, y_true_col: str) -> List[str]:
    if prob_cols and prob_cols != ["prob"]:
        classes = [c.replace("y_prob_", "") for c in prob_cols]
    else:
        classes = sorted(df[y_true_col].dropna().unique().tolist())
    return [str(c) for c in classes]


def compute_binary_metrics(
    y_true: np.ndarray,
    pos_prob: np.ndarray,
    threshold: Optional[float] = 0.5,
    auto_metric: str = "f1",
) -> Dict:
    """
    Compute precision/recall metrics for binary classification.
    If threshold is None or "auto", pick best threshold by specified metric over PR curve.
    """
    if threshold == "auto":
        precision, recall, thresh = precision_recall_curve(y_true, pos_prob)
        f1 = (2 * precision * recall) / np.clip(precision + recall, 1e-12, None)
        best_idx = int(np.nanargmax(f1))
        best_threshold = 0.5 if best_idx >= len(thresh) else float(thresh[best_idx])
    else:
        best_threshold = float(threshold)

    y_pred = (pos_prob >= best_threshold).astype(int)
    prec = precision_score(y_true, y_pred, zero_division=0)
    ap = average_precision_score(y_true, pos_prob)

    return {
        "threshold": best_threshold,
        "precision": float(prec),
        "average_precision": float(ap),
        "support": int(y_true.size),
    }


def compute_multiclass_metrics(
    y_true: np.ndarray,
    prob_mat: np.ndarray,
    classes: List[str],
    threshold: Optional[float] = 0.5,
    per_class: bool = True,
) -> Dict:
    y_pred_idx = np.argmax(prob_mat, axis=1)
    y_pred = np.array([classes[i] for i in y_pred_idx])
    overall_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)

    report = {
        "overall_precision_macro": float(overall_precision),
        "support": int(y_true.size),
    }

    if per_class:
        per_class_metrics = {}
        for j, cls in enumerate(classes):
            y_true_bin = (y_true == cls).astype(int)
            pos_prob = prob_mat[:, j]
            m = compute_binary_metrics(y_true_bin, pos_prob, threshold=threshold)
            per_class_metrics[cls] = m
        report["per_class"] = per_class_metrics

    try:
        report["classification_report"] = classification_report(y_true, y_pred, output_dict=False, zero_division=0)
    except Exception:
        report["classification_report"] = None

    return report


def plot_pr_curves(
    out_path: Path,
    y_true: np.ndarray,
    prob_source,
    classes: List[str],
) -> Optional[Path]:
    if not _HAS_PLT:
        return None

    out_path = out_path.with_suffix(".png")
    _ensure_dir(out_path)

    plt.figure()
    if prob_source.ndim == 1:
        precision, recall, _ = precision_recall_curve(y_true, prob_source)
        ap = average_precision_score(y_true, prob_source)
        plt.plot(recall, precision, label=f"AP={ap:.4f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall (Binary)")
        plt.legend(loc="best")
    else:
        for j, cls in enumerate(classes):
            y_true_bin = (y_true == cls).astype(int)
            precision, recall, _ = precision_recall_curve(y_true_bin, prob_source[:, j])
            ap = average_precision_score(y_true_bin, prob_source[:, j])
            plt.plot(recall, precision, label=f"{cls} (AP={ap:.4f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall (One-vs-Rest)")
        plt.legend(loc="best")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def run(
    artifacts_dir: Path,
    run_id: str,
    split: str,
    preds_path: Optional[Path],
    threshold: Optional[str | float],
    per_class: bool,
    plot: bool,
    verbose: bool,
    pos_label: str = "1",
) -> Dict:
    artifacts_dir = Path(artifacts_dir)
    if preds_path is None:
        preds_path = _default_preds_path(artifacts_dir, run_id, split)
    preds_path = Path(preds_path)

    if not preds_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {preds_path}")

    df = pd.read_csv(preds_path)
    y_true_col, prob_cols, y_pred_col = infer_label_columns(df)
    y_true = df[y_true_col].astype(str).values

    # Paths for outputs
    out_dir = artifacts_dir / run_id
    json_out = out_dir / f"precision_report_{split}.json"
    csv_out = out_dir / f"precision_report_{split}.csv"
    pr_png = out_dir / f"pr_curve_{split}.png"

    # Compute
    if prob_cols and is_binary(prob_cols, df, y_true_col):
        if prob_cols == ["prob"]:
            pos_prob = df["prob"].astype(float).values
        else:
            pos_candidates = [c for c in prob_cols if c.endswith("_1") or c.endswith("_positive") or c.endswith("_pos")]
            if len(pos_candidates) == 1:
                pos_prob = df[pos_candidates[0]].astype(float).values
            else:
                mat = df[prob_cols].astype(float).values
                if mat.shape[1] == 2:
                    pos_prob = mat.max(axis=1)
                else:
                    pos_prob = mat[:, 0]

        y_true_bin = (y_true == str(pos_label)).astype(int)

        m = compute_binary_metrics(
            y_true_bin,
            pos_prob,
            threshold=threshold,
        )

        report = {
            "task": "binary",
            "split": split,
            "metrics": m,
            "pos_label": str(pos_label),
        }

        if plot:
            plotted = plot_pr_curves(pr_png, y_true_bin, pos_prob, classes=["neg", "pos"])
            if plotted:
                report["pr_curve_path"] = str(plotted)

        csv_tbl = pd.DataFrame([{
            "split": split,
            "threshold": m["threshold"],
            "precision": m["precision"],
            "average_precision": m["average_precision"],
            "support": m["support"],
        }])
        _ensure_dir(csv_out)
        csv_tbl.to_csv(csv_out, index=False)

    else:
        if not prob_cols:
            raise ValueError("Multiclass analysis requires probability columns (y_prob_<class>).")

        classes = get_classes(prob_cols, df, y_true_col)
        prob_mat = df[prob_cols].astype(float).values
        row_sums = prob_mat.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            prob_mat = np.where(row_sums > 0, prob_mat / row_sums, prob_mat)

        m = compute_multiclass_metrics(y_true.astype(str), prob_mat, classes, threshold=threshold, per_class=per_class)

        report = {
            "task": "multiclass",
            "split": split,
            "classes": classes,
            "metrics": m,
        }

        if plot:
            plotted = plot_pr_curves(pr_png, y_true.astype(str), prob_mat, classes)
            if plotted:
                report["pr_curve_path"] = str(plotted)

        rows = []
        rows.append({
            "split": split,
            "class": "__overall_macro__",
            "threshold": "",
            "precision": m["overall_precision_macro"],
            "average_precision": "",
            "support": m["support"],
        })
        if per_class and "per_class" in m:
            for cls, cm in m["per_class"].items():
                rows.append({
                    "split": split,
                    "class": cls,
                    "threshold": cm.get("threshold", ""),
                    "precision": cm.get("precision", ""),
                    "average_precision": cm.get("average_precision", ""),
                    "support": cm.get("support", ""),
                })
        _ensure_dir(csv_out)
        pd.DataFrame(rows).to_csv(csv_out, index=False)

    _ensure_dir(json_out)
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logging.info("=== Precision Analysis (%s) ===", split)
    logging.info("Report JSON: %s", json_out)
    logging.info("Report CSV : %s", csv_out)
    if report.get("task") == "binary":
        m = report["metrics"]
        logging.info("Precision: %.4f  | AP: %.4f  | Thr: %s  | N: %d",
                     m["precision"], m["average_precision"], str(m["threshold"]), m["support"])
    else:
        m = report["metrics"]
        logging.info("Overall macro precision: %.4f | N: %d", m["overall_precision_macro"], m["support"])
        if per_class and "per_class" in m:
            for cls, cm in m["per_class"].items():
                logging.info("  [%s] Precision: %.4f | AP: %.4f | Thr: %s | N: %d",
                             cls, cm["precision"], cm["average_precision"], str(cm["threshold"]), cm["support"])

    return report


# -----------------------------
# CLI
# -----------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyze precision for a saved run's predictions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run-id", type=str, required=True, help="Run identifier (e.g., timestamped folder).")
    p.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"), help="Base artifacts directory.")
    p.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="Dataset split.")
    p.add_argument("--preds-path", type=Path, default=None, help="Optional direct path to predictions CSV.")
    p.add_argument(
        "--threshold",
        type=str,
        default="0.5",
        help='Decision threshold: float like "0.5" or "auto" (binary) to choose by max F1.',
    )
    p.add_argument("--pos-label", type=str, default="1", help="Positive label for binary reports (default: '1').")
    p.add_argument("--per-class", action="store_true", help="Emit per-class precision (multiclass).")
    p.add_argument("--plot", action="store_true", help="Save PR curve plot(s).")
    p.add_argument("--verbose", action="store_true", help="Verbose logging.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if args.threshold.lower() == "auto":
        thr: Optional[str | float] = "auto"
    else:
        try:
            thr = float(args.threshold)
        except ValueError:
            logging.error("Invalid --threshold value: %s (use a float or 'auto')", args.threshold)
            return 2

    try:
        run(
            artifacts_dir=args.artifacts_dir,
            run_id=args.run_id,
            split=args.split,
            preds_path=args.preds_path,
            threshold=thr,
            per_class=args.per_class,
            plot=args.plot,
            verbose=args.verbose,
            pos_label=args.pos_label,
        )
        return 0
    except Exception as e:
        logging.exception("Precision analysis failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
