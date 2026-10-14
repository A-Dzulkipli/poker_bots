#!/bin/bash
# Scratch: run.sh VERSION SEED PARALLEL OPP... ; one bench.py process per opponent, appended to logs/VERSION_sSEED.log
export UV_PROJECT_ENVIRONMENT=$HOME/p6venv
v=$1; s=$2; p=$3; shift 3
printf "%s\n" "$@" | xargs -P $p -I{} sh -c "uv run --frozen python bench.py $v:Bot {} --seed $s 2>&1 | grep bb/100 >> logs/${v}_s$s.log"
