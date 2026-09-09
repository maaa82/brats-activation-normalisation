#!/bin/bash
# =====================================================================
# Runs ON THE POD. Real-epoch cost benchmark: 12 activations x fold 0 x
# BENCH_EPOCHS epochs, basic nnU-Net (IN, batch 2), 5 at a time on the
# 5 A40s, each in its own tmux window (session "bench", windows named by
# activation). Waits for each wave to finish before the next so exactly
# five trainings share the CPU at any moment -- the same load as the
# production runs.
#
#   bash /workspace/run_bench12.sh            # 10 epochs each (default)
#   BENCH_EPOCHS=5 bash /workspace/run_bench12.sh
#
# ~15 min per wave at 10 epochs -> ~45 min total. Resumable: activations
# whose checkpoint_final.pth already exists are skipped.
# =====================================================================
set -u
export nnUNet_raw=/workspace/nnUNet_raw
export nnUNet_preprocessed=/workspace/nnUNet_preprocessed
export nnUNet_results=/workspace/nnUNet_results
export BENCH_EPOCHS="${BENCH_EPOCHS:-10}"
RES="$nnUNet_results/Dataset100_BraTS2023"
mkdir -p /workspace/logs /workspace/keep
ACTS=(LeakyReLU ReLU PReLU ELU GELU Swish Mish ELiSH HardELiSH TanhExp Logish Smish)

die() { printf '\n\033[31mSTOP: %s\033[0m\n' "$1"; exit 1; }

# --- preconditions -----------------------------------------------------
pgrep -f "nnunetv2.run.run_training" >/dev/null && die "training already running - GPUs must be idle"
TDIR=$(python3 -c "import nnunetv2,os;print(os.path.join(os.path.dirname(nnunetv2.__file__),'training','nnUNetTrainer'))")
[ -f "$TDIR/nnUNetTrainer_bench12.py" ] || { cp /workspace/nnUNetTrainer_bench12.py "$TDIR"/ || die "cannot install trainer"; }
python3 - <<'PY' || die "bench trainers do not resolve"
import os, nnunetv2
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
d=os.path.join(os.path.dirname(nnunetv2.__file__),"training","nnUNetTrainer")
bad=[a for a in "LeakyReLU ReLU PReLU ELU GELU Swish Mish ELiSH HardELiSH TanhExp Logish Smish".split()
     if recursive_find_python_class(d,f"nnUNetTrainer_bench_IN_{a}","nnunetv2.training.nnUNetTrainer") is None]
print("unresolved:", bad or "none"); raise SystemExit(1 if bad else 0)
PY
[ "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" -ge 5 ] || die "need 5 GPUs"
command -v tmux >/dev/null || die "tmux missing"
tmux has-session -t bench 2>/dev/null && die "tmux session 'bench' exists - kill it first: tmux kill-session -t bench"

done_for() { [ -f "$RES/nnUNetTrainer_bench_IN_$1__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth" ]; }
TODO=(); for a in "${ACTS[@]}"; do done_for "$a" && echo "  $a already done - skip" || TODO+=("$a"); done
[ ${#TODO[@]} -gt 0 ] || { echo "all 12 already benchmarked"; exit 0; }
echo "to run (${#TODO[@]}): ${TODO[*]}   BENCH_EPOCHS=$BENCH_EPOCHS   $(date)"

ENV="export nnUNet_raw=$nnUNet_raw nnUNet_preprocessed=$nnUNet_preprocessed nnUNet_results=$nnUNet_results BENCH_EPOCHS=$BENCH_EPOCHS"
wave=0
for ((i=0; i<${#TODO[@]}; i+=5)); do
  wave=$((wave+1)); batch=("${TODO[@]:i:5}")
  echo; echo "=== wave $wave: ${batch[*]}   $(date) ==="
  g=0
  for a in "${batch[@]}"; do
    TR="nnUNetTrainer_bench_IN_$a"; LOG="/workspace/logs/bench_${a}.out"
    CMD="$ENV; CUDA_VISIBLE_DEVICES=$g python3 -u -m nnunetv2.run.run_training 100 3d_fullres 0 -tr $TR --c 2>&1 | tee -a $LOG; echo '===== $a finished exit \${PIPESTATUS[0]} $(date) ====='; sleep 3600"
    if ! tmux has-session -t bench 2>/dev/null; then tmux new-session -d -s bench -n "$a" "$CMD"
    else tmux new-window -t bench -n "$a" "$CMD"; fi
    echo "  GPU $g -> $a   (tmux bench:$a)"; g=$((g+1)); sleep 2
  done
  # wait for this wave
  sleep 90
  while pgrep -f "nnunetv2.run.run_training" >/dev/null; do sleep 30; done
  for a in "${batch[@]}"; do done_for "$a" && echo "  $a: done" || echo "  $a: NO checkpoint_final - check tmux bench:$a"; done
done

echo; echo "=== all waves finished $(date) ==="
echo "parse with:  python3 /workspace/parse_bench12.py"
echo "tmux windows stay open 1 h for inspection; kill with: tmux kill-session -t bench"
