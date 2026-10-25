"""
Automated Retraining Pipeline
=================================

Runs the full chain (pull fresh MT5 data -> relabel -> rebuild features ->
retrain) for one instrument, with a safety gate: the new model is only
swapped into production after basic checks pass, and the old model is
backed up rather than overwritten blind.

This is NOT reinforcement learning - RL would try to learn a trading POLICY
through trial-and-error exploration, which needs far more data than a young
live account has and is notoriously unstable in low-signal environments
like this. This is much simpler and safer: periodically re-run the SAME
validated supervised-learning pipeline on an extended historical window,
exactly like every retrain done manually so far - just automated.

IMPORTANT: this refreshes the model using extended HISTORICAL data (fresh
MT5 pull up to today). It does NOT fold your actual live trade outcomes
into training - that's a separate, more involved step not built here.

For TRUE periodic automation, schedule it with Windows Task Scheduler - see
the printed instructions at the end of a successful run.

Usage:
    python retrain_pipeline.py --symbol XAUUSD --out-root phase1_output --embargo-bars 48
    python retrain_pipeline.py --symbol NAS100 --out-root phase1_output_nas100 --embargo-bars 48 --calibration-method platt
"""

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_step(description, cmd):
    print(f"\n{'='*70}\n{description}\n{'='*70}")
    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout[-3000:])
    if result.returncode != 0:
        print(f"FAILED: {description}")
        print(result.stderr[-2000:])
        return False
    return True


def run(symbol, out_root, embargo_bars, calibration_method, min_sanity_auc):
    out_root = Path(out_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Retraining pipeline for {symbol} - started {timestamp}")

    if not run_step(
        f"Step 1/4: Pull fresh {symbol} data",
        [sys.executable, "src/mltrader/data/nas100_phase1_pull.py", "--symbol", symbol, "--out", str(out_root)]
    ):
        return

    if not run_step(
        "Step 2/4: Triple-barrier labeling",
        [sys.executable, "src/mltrader/labeling/triple_barrier.py",
         "--events", str(out_root / "events" / "M15_events.csv"),
         "--out", str(out_root / "labels")]
    ):
        return

    if not run_step(
        "Step 3/4: Feature engineering",
        [sys.executable, "src/mltrader/features/features.py",
         "--m15-events", str(out_root / "events" / "M15_events.csv"),
         "--h1-events", str(out_root / "events" / "H1_events.csv"),
         "--h4-events", str(out_root / "events" / "H4_events.csv"),
         "--labels", str(out_root / "labels" / "M15_triple_barrier_labels.csv"),
         "--out", str(out_root / "features")]
    ):
        return

    staging_dir = out_root / f"production_model_staging_{timestamp}"
    if not run_step(
        "Step 4/4: Train new model (to staging)",
        [sys.executable, "src/mltrader/models/train_final_model.py",
         "--dataset", str(out_root / "features" / "M15_feature_dataset.csv"),
         "--out", str(staging_dir),
         "--embargo-bars", str(embargo_bars),
         "--calibration-method", calibration_method]
    ):
        return

    model_file = staging_dir / "xgb_model.joblib"
    calibrator_file = staging_dir / "calibrator.joblib"
    if not model_file.exists() or not calibrator_file.exists():
        print("\nSAFETY GATE FAILED: expected model files not found in staging. "
              "NOT swapping into production. Investigate the training step above.")
        return
    if model_file.stat().st_size < 1000 or calibrator_file.stat().st_size < 100:
        print("\nSAFETY GATE FAILED: model/calibrator files are suspiciously small. "
              "NOT swapping into production.")
        return

    print("\nSAFETY GATE PASSED (basic file checks). "
          "NOTE: this does NOT re-run the full OOS validation - that's still a "
          "manual step you should do periodically, not something this script "
          "verifies automatically.")

    production_dir = out_root / "production_model"
    if production_dir.exists():
        backup_dir = out_root / f"production_model_backup_{timestamp}"
        shutil.copytree(production_dir, backup_dir)
        print(f"\nBacked up old production model to: {backup_dir}")
        shutil.rmtree(production_dir)

    shutil.move(str(staging_dir), str(production_dir))
    print(f"New model promoted to: {production_dir}")

    print("\n" + "=" * 70)
    print("RETRAIN COMPLETE")
    print("=" * 70)
    print(f"Symbol: {symbol}")
    print(f"New production model: {production_dir}")
    print("\nIMPORTANT: restart live_runner_multi.py to pick up the new model -")
    print("it only loads the model at startup, not automatically mid-run.")
    print("\nAlso recommended (not automated here): re-run realistic_backtest.py")
    print("with --exclude-recent-months on this refreshed dataset to confirm the")
    print("new model still validates on a genuinely untouched window before")
    print("fully trusting it.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated retraining pipeline")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--embargo-bars", type=int, default=48)
    parser.add_argument("--calibration-method", choices=["isotonic", "platt"], default="isotonic")
    parser.add_argument("--min-sanity-auc", type=float, default=0.52)
    args = parser.parse_args()

    run(args.symbol, args.out_root, args.embargo_bars, args.calibration_method,
        args.min_sanity_auc)
