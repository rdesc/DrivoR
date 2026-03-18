#!/bin/bash
# Launch all three training runs, each as a chain of N jobs so the total
# wall-time budget covers 10 epochs (~10 hours) within the 3-hour
# short-unkillable partition limit.
#
# Usage:
#   bash launch_all_runs.sh          # launches all 3 runs
#   bash launch_all_runs.sh grpo     # launches only the GRPO run
#   bash launch_all_runs.sh scratch  # launches only the from-scratch run
#   bash launch_all_runs.sh baseline # launches only the BCE baseline run
#
# Each job picks up from last.ckpt automatically (see individual sbatch scripts).

NUM_JOBS=5   # 5 x 3h = 15h budget for 10 epochs (~1h/epoch)

chain_jobs() {
    local script=$1
    local label=$2
    local prev_jid=""

    echo "=== Launching $NUM_JOBS chained jobs for: $label ==="
    for i in $(seq 1 $NUM_JOBS); do
        if [ -z "$prev_jid" ]; then
            jid=$(sbatch --parsable "$script")
        else
            jid=$(sbatch --parsable --dependency=afterany:$prev_jid "$script")
        fi
        echo "  Job $i: $jid (depends on: ${prev_jid:-none})"
        prev_jid=$jid
    done
    echo ""
}

RUN=${1:-all}

if [ "$RUN" = "all" ] || [ "$RUN" = "grpo" ]; then
    chain_jobs train_drivor_mila.sh            "GRPO Option 3"
fi

if [ "$RUN" = "all" ] || [ "$RUN" = "scratch" ]; then
    chain_jobs train_drivor_scratch_mila.sh    "DrivoR from scratch"
fi

if [ "$RUN" = "all" ] || [ "$RUN" = "baseline" ]; then
    chain_jobs train_drivor_bce_baseline_mila.sh "BCE baseline (frozen backbone)"
fi
