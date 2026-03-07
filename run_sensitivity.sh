#!/bin/bash
# Targeted sensitivity analysis: 21 runs
set -e

export DATA_ROOT="/Users/antemilicevic/Documents/Project/code/Data"
REPO="/Users/antemilicevic/Documents/Project/final-repo"
RESULTS_DIR="$REPO/sensitivity_results"
RESULTS_CSV="$RESULTS_DIR/all_results.csv"
OUTPUT_BASE="$REPO/final/outputs"

mkdir -p "$RESULTS_DIR/details"

# CSV header
echo "run_id,group,walk_length,num_walks,window,p,q,male_icd_mean_drift,male_icd_mean_stab,male_icd_mean_auc,male_blk_mean_drift,male_blk_mean_stab,male_blk_mean_auc,fem_icd_mean_drift,fem_icd_mean_stab,fem_icd_mean_auc,fem_blk_mean_drift,fem_blk_mean_stab,fem_blk_mean_auc,male_icd_ni_drift,male_icd_ni_stab,male_blk_ni_drift,male_blk_ni_stab,fem_icd_ni_drift,fem_icd_ni_stab,fem_blk_ni_drift,fem_blk_ni_stab,runtime_s" > "$RESULTS_CSV"

mean_col() {
    local file="$1" col="$2"
    [ ! -f "$file" ] && echo "NA" && return
    local idx=$(head -1 "$file" | tr ',' '\n' | grep -n "^${col}$" | cut -d: -f1)
    [ -z "$idx" ] && echo "NA" && return
    awk -F',' -v c="$idx" 'NR>1{s+=$c;n++}END{if(n>0)printf "%.6f",s/n;else print "NA"}' "$file"
}
mean_auc() {
    local file="$1"
    [ ! -f "$file" ] && echo "NA" && return
    local idx=$(head -1 "$file" | tr ',' '\n' | grep -n "^auc$" | cut -d: -f1)
    [ -z "$idx" ] && echo "NA" && return
    awk -F',' -v c="$idx" 'NR>1 && $c!="" && $c!="nan" && $c!="NaN"{s+=$c;n++}END{if(n>0)printf "%.6f",s/n;else print "NA"}' "$file"
}

run_one() {
    local RID="$1" GRP="$2" WL="$3" NW="$4" WIN="$5" P="$6" Q="$7"
    local TAG="r${RID}_wl${WL}_nw${NW}_w${WIN}_p${P}_q${Q}"
    echo ""
    echo "======== RUN $RID/21 [$GRP]: wl=$WL nw=$NW w=$WIN p=$P q=$Q ========"
    local T0=$(date +%s)
    cd "$REPO"
    python final/run_all_combinations.py --dim 128 --walk-length "$WL" --num-walks "$NW" --window "$WIN" --p "$P" --q "$Q" --seed 42 --force 2>&1 | grep -E "INFO.*(DONE|SUMMARY|All done)" || true
    local T1=$(date +%s)
    local RT=$((T1-T0))

    # Collect metrics
    local mi_d=$(mean_col "$OUTPUT_BASE/tables/male/icd/drift_summary.csv" "mean_drift")
    local mi_s=$(mean_col "$OUTPUT_BASE/tables/male/icd/stability_summary.csv" "mean_stability")
    local mi_a=$(mean_auc "$OUTPUT_BASE/tables/male/icd/structure_preservation.csv")
    local mb_d=$(mean_col "$OUTPUT_BASE/tables/male/blocks/drift_summary.csv" "mean_drift")
    local mb_s=$(mean_col "$OUTPUT_BASE/tables/male/blocks/stability_summary.csv" "mean_stability")
    local mb_a=$(mean_auc "$OUTPUT_BASE/tables/male/blocks/structure_preservation.csv")
    local fi_d=$(mean_col "$OUTPUT_BASE/tables/female/icd/drift_summary.csv" "mean_drift")
    local fi_s=$(mean_col "$OUTPUT_BASE/tables/female/icd/stability_summary.csv" "mean_stability")
    local fi_a=$(mean_auc "$OUTPUT_BASE/tables/female/icd/structure_preservation.csv")
    local fb_d=$(mean_col "$OUTPUT_BASE/tables/female/blocks/drift_summary.csv" "mean_drift")
    local fb_s=$(mean_col "$OUTPUT_BASE/tables/female/blocks/stability_summary.csv" "mean_stability")
    local fb_a=$(mean_auc "$OUTPUT_BASE/tables/female/blocks/structure_preservation.csv")
    local mi_ni_d=$(mean_col "$OUTPUT_BASE/tables/male/icd/drift_summary_non_isolates.csv" "mean_drift")
    local mi_ni_s=$(mean_col "$OUTPUT_BASE/tables/male/icd/stability_summary_non_isolates.csv" "mean_stability")
    local mb_ni_d=$(mean_col "$OUTPUT_BASE/tables/male/blocks/drift_summary_non_isolates.csv" "mean_drift")
    local mb_ni_s=$(mean_col "$OUTPUT_BASE/tables/male/blocks/stability_summary_non_isolates.csv" "mean_stability")
    local fi_ni_d=$(mean_col "$OUTPUT_BASE/tables/female/icd/drift_summary_non_isolates.csv" "mean_drift")
    local fi_ni_s=$(mean_col "$OUTPUT_BASE/tables/female/icd/stability_summary_non_isolates.csv" "mean_stability")
    local fb_ni_d=$(mean_col "$OUTPUT_BASE/tables/female/blocks/drift_summary_non_isolates.csv" "mean_drift")
    local fb_ni_s=$(mean_col "$OUTPUT_BASE/tables/female/blocks/stability_summary_non_isolates.csv" "mean_stability")

    echo "$RID,$GRP,$WL,$NW,$WIN,$P,$Q,$mi_d,$mi_s,$mi_a,$mb_d,$mb_s,$mb_a,$fi_d,$fi_s,$fi_a,$fb_d,$fb_s,$fb_a,$mi_ni_d,$mi_ni_s,$mb_ni_d,$mb_ni_s,$fi_ni_d,$fi_ni_s,$fb_ni_d,$fb_ni_s,$RT" >> "$RESULTS_CSV"

    # Save detail files
    local DDIR="$RESULTS_DIR/details/$TAG"
    mkdir -p "$DDIR"
    for sex in male female; do
      for nt in icd blocks chronic; do
        mkdir -p "$DDIR/${sex}_${nt}"
        cp "$OUTPUT_BASE/tables/$sex/$nt/drift_summary.csv" "$DDIR/${sex}_${nt}/" 2>/dev/null || true
        cp "$OUTPUT_BASE/tables/$sex/$nt/stability_summary.csv" "$DDIR/${sex}_${nt}/" 2>/dev/null || true
        cp "$OUTPUT_BASE/tables/$sex/$nt/structure_preservation.csv" "$DDIR/${sex}_${nt}/" 2>/dev/null || true
        cp "$OUTPUT_BASE/tables/$sex/$nt/drift_summary_non_isolates.csv" "$DDIR/${sex}_${nt}/" 2>/dev/null || true
        cp "$OUTPUT_BASE/tables/$sex/$nt/stability_summary_non_isolates.csv" "$DDIR/${sex}_${nt}/" 2>/dev/null || true
        cp "$OUTPUT_BASE/tables/$sex/$nt/top_drifters.csv" "$DDIR/${sex}_${nt}/" 2>/dev/null || true
      done
    done
    echo "  Done in ${RT}s — saved to $DDIR"
}

# 0: Baseline
run_one 1 baseline 80 10 10 1.0 1.0

# 1: Walk strategy (p,q) experiments
run_one 2  walk_strategy 80 10 8  1.0 1.0
run_one 3  walk_strategy 80 10 12 1.0 1.0
run_one 4  walk_strategy 80 10 8  0.5 2.0
run_one 5  walk_strategy 80 10 12 0.5 2.0
run_one 6  walk_strategy 80 10 8  2.0 0.5
run_one 7  walk_strategy 80 10 12 2.0 0.5

# 2: Walk coverage experiments
run_one 8  walk_coverage 60  10 10 1.0 1.0
run_one 9  walk_coverage 80  10 10 1.0 1.0
run_one 10 walk_coverage 120 10 10 1.0 1.0
run_one 11 walk_coverage 80  20 10 1.0 1.0
run_one 12 walk_coverage 120 20 10 1.0 1.0
run_one 13 walk_coverage 60  20 10 1.0 1.0

# 3: Combined experiments
run_one 14 combined 120 20 10 0.5 2.0
run_one 15 combined 120 20 10 2.0 0.5
run_one 16 combined 120 20 8  0.5 2.0
run_one 17 combined 120 20 12 0.5 2.0
run_one 18 combined 120 20 8  2.0 0.5
run_one 19 combined 120 20 12 2.0 0.5

# Note: runs 1 and 9 are identical (baseline), so 20 unique configs + 1 duplicate = 21 runs
# Keeping both as requested for completeness

echo ""
echo "======== ALL 21 RUNS COMPLETE ========"
echo "Results: $RESULTS_CSV"
