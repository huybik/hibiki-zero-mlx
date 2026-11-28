#!/bin/bash
# The slices student's finish once its on-policy run (both_slices_z) has stopped and synced
# (slices-on-policy.md#phone-finish), each step after the one before, on this checkout:
#   1. the slices' refit on the on-policy best (restart.sh student-slices-refit-5090), its final
#      model hard-linked to weights/both_slices_z_refit.safetensors;
#   2. the 6-bit finish from it (restart.sh student-slices-pal6-5090);
#   3. the refit and the finish's final update snapped onto their 6-bit tables (training.student.palettize);
#   4. both snapped models uploaded to the Hub repo's phone/ (the Mac's export fetches them there:
#      ssh to the Mac runs at ~2 MB/s, the Hub at ~30);
#   5. test_student.sh on the on-policy start, the refit (bf16 and snapped) and the finish (snapped),
#      under runs/slices_tests/.
# A step whose output exists is skipped (an upload the Hub already holds sends nothing), so a rerun
# continues. Run it detached:
#   setsid nohup training/tranche/box/finish_slices.sh >> /workspace/logs/finish_slices.log 2>&1 < /dev/null &
set -euo pipefail
cd /workspace/hibiki-zero
L=/workspace/logs B=training/tranche/box C=weights/student_slices14/config.json O=runs/slices_tests
log() { echo "$(date -u '+%F %T') $*"; }
exec 9> $L/finish_slices.lock
flock -n 9 || { log "another finish_slices.sh holds $L/finish_slices.lock"; exit 1; }
run_job() {  # restart.sh job $1, waited for, unless its final model $2 exists; stops the chain without it
  [[ -f $2 ]] && return
  log "start $1"
  $B/restart.sh "$1" 9>&-  # the detached job must not hold the chain's lock
  sleep 5
  local g; g=$(cat $L/$1.pid)
  while kill -0 "$g" 2>/dev/null; do sleep 30; done  # h100.sh, the group's leader, exits after its sync
  kill -- -"$g" 2>/dev/null || true  # workers a run killed out of memory leaves in its group (restart.sh)
  log "end $1"
  [[ -f $2 ]] || { log "$1 ended without $2: see $L/$1.log"; exit 1; }
}
snap() { [[ -f $2 ]] || { log "snap $1"; .venv/bin/python -m training.student.palettize --weight "$1" --bits 6 --out "$2"; }; }
publish() {
  log "upload $1"
  (set -a; . /workspace/.env; set +a; HF_HUB_DISABLE_PROGRESS_BARS=1 .venv/bin/python -c 'import sys, os; from huggingface_hub import HfApi
HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo="phone/" + os.path.basename(sys.argv[1]),
                    repo_id="huybik/hibiki-zero-vi-full-sft")' "$1")
}

u=$(.venv/bin/python -c 'import json; print(json.load(open("runs/both_slices_z/best.json"))["step"])')
# Every output below but the refit's run is named for no update: they belong to the best they started from.
F=runs/slices_finish_best.txt
[[ -f $F ]] || echo "$u" > $F
[[ $(cat $F) == "$u" ]] || { log "the outputs are from best $(cat $F), now $u: move them aside and $F away"; exit 1; }
R=runs/slices_z${u}_refit
run_job student-slices-refit-5090 $R/model_step002000.safetensors
ln -f $R/model_step002000.safetensors weights/both_slices_z_refit.safetensors
P=runs/pal6_slices_z
run_job student-slices-pal6-5090 $P/model_step004000.safetensors
snap weights/both_slices_z_refit.safetensors weights/both_slices_z_refit_snap6.safetensors
snap $P/model_step004000.safetensors weights/pal6_slices_z4000_snap6.safetensors
publish weights/both_slices_z_refit_snap6.safetensors
publish weights/pal6_slices_z4000_snap6.safetensors
for t in "student_slices14_decay8000 decay8000" "both_slices_z_refit refit" \
         "both_slices_z_refit_snap6 refit_snap6" "pal6_slices_z4000_snap6 pal6_4000_snap6"; do
  set -- $t
  log "test $2"
  $B/test_student.sh $C weights/$1.safetensors $O/$2
done
log "done"
