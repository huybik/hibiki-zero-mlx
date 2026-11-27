#!/bin/bash
# The slices student's finish once its on-policy run (both_slices_z) has stopped and synced
# (slices-on-policy.md#phone-finish), each step after the one before, on this checkout:
#   1. the slices' refit on the on-policy best (restart.sh student-slices-refit-5090), its final
#      model hard-linked to weights/both_slices_z_refit.safetensors;
#   2. the 6-bit finish from it (restart.sh student-slices-pal6-5090);
#   3. the refit and the finish's final update snapped onto their 6-bit tables (training.student.palettize);
#   4. test_student.sh on the on-policy start, the refit (bf16 and snapped) and the finish (snapped),
#      under runs/slices_tests/.
# A step whose output exists is skipped, so a rerun continues. Run it detached:
#   setsid nohup training/tranche/box/finish_slices.sh >> /workspace/logs/finish_slices.log 2>&1 < /dev/null &
set -euo pipefail
cd /workspace/hibiki-zero
L=/workspace/logs B=training/tranche/box C=weights/student_slices14/config.json O=runs/slices_tests
log() { echo "$(date -u '+%F %T') $*"; }
run_job() {  # a restart.sh job, waited for
  log "start $1"
  $B/restart.sh "$1"
  sleep 5
  while kill -0 -- -"$(cat $L/$1.pid)" 2>/dev/null; do sleep 30; done
  log "end $1"
}
snap() { [[ -f $2 ]] || { log "snap $1"; .venv/bin/python -m training.student.palettize --weight "$1" --bits 6 --out "$2"; }; }

u=$(.venv/bin/python -c 'import json; print(json.load(open("runs/both_slices_z/best.json"))["step"])')
R=runs/slices_z${u}_refit
[[ -f $R/model_step002000.safetensors ]] || run_job student-slices-refit-5090
ln -f $R/model_step002000.safetensors weights/both_slices_z_refit.safetensors
P=runs/pal6_slices_z
[[ -f $P/model_step004000.safetensors ]] || run_job student-slices-pal6-5090
snap weights/both_slices_z_refit.safetensors weights/both_slices_z_refit_snap6.safetensors
snap $P/model_step004000.safetensors weights/pal6_slices_z4000_snap6.safetensors
for t in "student_slices14_decay8000 decay8000" "both_slices_z_refit refit" \
         "both_slices_z_refit_snap6 refit_snap6" "pal6_slices_z4000_snap6 pal6_4000_snap6"; do
  set -- $t
  log "test $2"
  $B/test_student.sh $C weights/$1.safetensors $O/$2
done
log "done"
