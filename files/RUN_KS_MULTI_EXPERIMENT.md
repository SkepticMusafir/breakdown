# Running the ks_multi Experiment

## Status
✅ **Code ready in notebook** — `baseline_with_viz.ipynb` cell 86f0a0f5 has `SCORING_METHOD = "ks_multi"`
⏳ **Awaiting execution** — Requires interactive Jupyter with crunch environment

## What Was Changed
Cell `86f0a0f5` in `baseline_with_viz.ipynb`:
```python
SCORING_METHOD = "ks_multi"   # Changed from "ewma"
ALPHA = 0.05
KAPPA = 3.0
KAPPA_KS = 3.7
```

This enables the three-view KS-test scoring method:
- **View 1 (raw values)**: Catches location and shape shifts
- **View 2 (absolute deviation)**: Catches variance/scale shifts
- **View 3 (first differences)**: Catches autocorrelation/structure shifts
- **Score**: `max(s1, s2, s3)` — takes the strongest signal across all views

## How to Run

### Prerequisites
- Jupyter notebook installed with crunch environment set up
- Valid crunch token/project context (`.crunch` file)
- Internet access to download data (on first run)

### Step-by-Step Execution
1. Open `baseline_with_viz.ipynb` in Jupyter Lab or Notebook
2. Verify cell 86f0a0f5 shows `SCORING_METHOD = "ks_multi"` (it should after the recent edit)
3. Run cells sequentially from the top:
   - Dependencies cell
   - `crunch_tools = crunch.load_notebook()` cell
   - `train_data, test_data = crunch_tools.load_data()` cell  
   - **Run the `crunch_tools.test()` cell** ← This is the key step
4. Wait ~5-10 minutes for execution
5. Look for output: `Local TS-AUC: X.XXXX` in the "Computing TS-AUC locally" cell

### Alternative: Command Line
If you have crunch CLI configured:
```bash
cd d:\breakdown_1\files
crunch test --notebook baseline_with_viz.ipynb
```

## Expected Output
When `crunch_tools.test()` runs:
```
05:25:32 running local test
05:25:33 executing - command=train
05:25:49 executing - command=infer
[parallel execution across 4 cores]
05:25:51 determinism check: passed
05:25:51 save prediction - path=prediction

Local TS-AUC: X.XXXX
```

## Recording the Result
After execution:
1. Note the **Local TS-AUC** value
2. Update EXPERIMENTS.md row 4:
   - Change `**awaiting execution**` to the actual value (e.g., `**0.5234**`)
   - Update notes if needed
3. Commit the updated EXPERIMENTS.md

## Diagnostic Checks
If TS-AUC is still ~0.5 (essentially random), run these checks in the notebook:
```python
# Check 1: Score variance collapse?
print(f"Prediction std: {merged['prediction'].std():.6f}")  # Should NOT be < 0.01

# Check 2: Index alignment?
print(f"Merged shape: {merged.shape}")
print(f"y_test shape: {y_test.shape}")  # Should be close

# Check 3: Compare to EWMA baseline
# Run row 1 experiment (SCORING_METHOD = "ewma") and compare TS-AUC
```

If these checks show issues, see **Open issue §5** in EXPERIMENTS.md.
