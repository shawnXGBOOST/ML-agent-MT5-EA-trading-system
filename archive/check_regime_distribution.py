"""Quick diagnostic — run after regime.py to sanity-check the regime buckets
aren't degenerate (e.g. everything clustered in one bucket, which would
trivially explain flat expectancy across regimes)."""

import argparse
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime-dataset", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.regime_dataset)

    print("ADX(14) distribution:")
    print(df["adx_14"].describe())
    print()
    print("trend_regime counts:")
    print(df["trend_regime"].value_counts())
    print()
    print("vol_regime counts:")
    print(df["vol_regime"].value_counts())
    print()
    print("If any single bucket holds >80% of rows, or ADX std is very small "
          "relative to its mean, the regime split isn't meaningfully differentiating "
          "conditions — that would explain flat expectancy across regimes.")
