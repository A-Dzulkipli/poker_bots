#!/bin/bash
# Scratch: run one candidate against several opponents in parallel (foreground); append to logs/<cand>_s<seed>.log
cand=$1; seed=$2; shift 2
export UV_PROJECT_ENVIRONMENT=$HOME/p4venv
for opp in "$@"; do uv run --frozen python bench.py $cand $opp --seed $seed 2>&1 | grep " vs " >> logs/${cand%%:*}_s${seed}.log & done
wait
