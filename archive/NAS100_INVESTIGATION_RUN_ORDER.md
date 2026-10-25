# NAS100 Investigation — Run Order

Four new files. Run them in this order; each one's result decides whether the
next is worth doing.

---

## Step 1 — Distribution shift diagnostic (run this first)

```
python distribution_shift.py --dataset phase1_output_nas100\features\M15_feature_dataset.csv --split-date 2026-03-11 --out phase1_output_nas100\distribution_shift
```

This is the single most decisive test we have left, and it's cheap. It answers
three questions at once:

1. **Are the model's inputs still familiar?** (feature PSI — <0.10 = stable)
2. **Did the regime mix change?** (frequency comparison)
3. **THE KEY ONE: within each individual regime, did the win rate itself drop?**

That third table decides everything:

- **Regime mix changed, but per-regime win rates held steady** → the strategy
  didn't break. The window just contained more of the conditions NAS100 was
  always weak in. Regime exclusion (already applied in `agent.py`) is the
  correct fix, and no further work is needed.

- **Per-regime win rates themselves dropped** → something structural changed.
  Regime filtering won't save it, and Step 2 becomes worth doing.

Run the same thing for gold as a control, so you can see what "healthy" looks
like for comparison:

```
python distribution_shift.py --dataset phase1_output\features\M15_feature_dataset.csv --split-date 2026-03-11 --out phase1_output\distribution_shift
```

---

## Step 2 — Session features (only if Step 1 shows per-regime degradation)

The idea (from the ChatGPT review, and I think it's their strongest point):
for an index CFD anchored to the US cash session, "BOS at 09:35" and "BOS at
14:20" are genuinely different events, and our ADX+ATR feature set literally
cannot tell them apart. Gold trades ~24h with no single dominant session, so
this matters much less there.

Adds 15 features: overnight range, opening gap, distance from previous US
cash close, opening-range breakout distances, and session-phase flags
(pre-market / opening 30m / morning / lunch / afternoon / post-close).

```
python features_session.py --m15-events phase1_output_nas100\events\M15_events.csv --h1-events phase1_output_nas100\events\H1_events.csv --h4-events phase1_output_nas100\events\H4_events.csv --labels phase1_output_nas100\labels\M15_triple_barrier_labels.csv --out phase1_output_nas100\features_session
```

Then the A/B test that tells you whether they actually help:

```
python compare_feature_sets.py --dataset phase1_output_nas100\features_session\M15_feature_dataset_session.csv --out phase1_output_nas100\feature_comparison --n-folds 5 --embargo-bars 48
```

Same model, same purged CV, only the feature set differs. It reports the AUC
change, how many folds improved, and what share of feature importance the new
features earned.

**A negative result here is genuinely useful** — it rules out "missing session
structure" as the explanation and stops you sinking more time into that
direction.

I verified this runs correctly, but only on synthetic data, where the apparent
improvement was noise. Your real data is the actual test.

---

## Step 3 — `agent.py` is already updated (no command needed)

NAS100's three confirmed-weak regimes are now **excluded outright**, not just
given a higher EV threshold:

- `trending_low_vol` (20-27% win rate across four independently-trained models)
- `ranging_high_vol` (27-32%)
- `ranging_normal_vol` (27-31%)

Why exclusion rather than a higher threshold: we tested the threshold approach
(0.25-0.40) and it barely moved the win rate (38.8% vs 37.5-40.5% before).
That indicates the EV estimate in those regimes is uniformly overoptimistic
rather than just needing a higher cut-off — so filtering on it doesn't sort
good trades from bad.

Gold is unaffected: its thresholds are unchanged, and gold's `ranging_high_vol`
still trades normally. The exclusions are keyed by (symbol, regime).

Re-run the combined backtest to see the effect:

```
python realistic_backtest_multi.py --xauusd-model-dir phase1_output\oos_test_model --nas100-model-dir phase1_output_nas100\oos_test_model --months 6 --starting-equity 800 --out phase1_output\oos_multi_v3 --xauusd-spread-price 0.30 --xauusd-commission-price 0.05 --nas100-spread-price 1.0 --nas100-commission-price 0.5
```

Expect NAS100's trade count to drop meaningfully (those three regimes were a
large share of its signals) and its win rate to rise. If the win rate does NOT
rise after removing regimes that were running 20-32%, that itself is a red flag
worth telling me about — it would mean the regime labels aren't capturing what
we think they are.

---

## Where things actually stand

**XAUUSD: validated and production-ready.** 44.2% OOS win rate on a genuinely
unseen 6-month window, matching its 44.5-47.7% benchmark. Three independent
checks now agree. This one you can run live with reasonable confidence.

**NAS100: still under research, not production-ready.** Persistent 37-40% vs a
45.1% benchmark. We've ruled out: cost-model error, concurrency selection
effects, entry-price convention, calibration drift by direction, and stale
training data (four differently-windowed models all landed in the same band).
What's left is either a composition effect (Step 1's third table will say) or
something structural in how BOS events are defined for an index.

My honest recommendation: **run gold live now, keep NAS100 in research.** Don't
let an unresolved instrument hold back the one that's actually working, and
don't put money behind NAS100 until its numbers stop disagreeing with its own
validation.
