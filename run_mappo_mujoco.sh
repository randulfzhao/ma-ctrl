#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${ROOT_DIR}/run_mappo"
cd "${ROOT_DIR}"

# Explicit script list (ordered). Maintain this list manually.
scripts=(
  "ant2x4_0.sh"
  "ant2x4_1.sh"
  "ant2x4d_0.sh"
  "ant2x4d_1.sh"
  "ant4x2_0.sh"
  "ant4x2_1.sh"
  "cheetah6x1_0.sh"
  "cheetah6x1_1.sh"
  "swimmer_0.sh"
  "swimmer_1.sh"
  "walker_0.sh"
  "walker_1.sh"
)

if [[ ${#scripts[@]} -eq 0 ]]; then
  echo "No scripts configured."
  exit 0
fi

echo "Found ${#scripts[@]} script(s):"
printf '  %s\n' "${scripts[@]}"
echo

declare -a pids
declare -a names
overall=0

for script in "${scripts[@]}"; do
  script_path="${RUN_DIR}/${script}"
  if [[ ! -f "$script_path" ]]; then
    echo "Skip $script (not found)"
    overall=1
    continue
  fi
  # Enable pipefail in child scripts so `python ... | tee ...` failures propagate.
  bash -o pipefail "$script_path" &
  pid=$!
  pids+=("$pid")
  names+=("$script")
  echo "Started $script (PID: $pid)"
done

echo
echo "Waiting for all scripts to finish..."

for i in "${!pids[@]}"; do
  pid="${pids[$i]}"
  name="${names[$i]}"
  if wait "$pid"; then
    code=0
  else
    code=$?
    overall=1
  fi
  echo "Finished $name (exit code: $code)"
done

exit "$overall"
