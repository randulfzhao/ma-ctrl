#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${ROOT_DIR}/run_enode"
cd "${ROOT_DIR}"

# Explicit script list (ordered). Maintain this list manually.
scripts=(
  "cooperative_predator_prey_0.sh"
  "cooperative_predator_prey_1.sh"
  "cooperative_navigation_0.sh"
  "cooperative_navigation_1.sh"
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
  bash "$script_path" &
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
