#!/usr/bin/env bash
# Publish a CI log file to the throwaway `ci-logs` branch so a failure can be
# read without authenticated access to the workflow log (raw.githubusercontent
# and the contents API serve that branch publicly).
#
#   GITHUB_TOKEN=... scripts/ci/publish-log.sh <log-file> <published-name>
#
# Appends to the branch instead of force-pushing an orphan: several jobs in one
# run publish concurrently, and an orphan push from the last job used to delete
# every other job's log. Retries on a non-fast-forward push.
set -euo pipefail
# Never wait on a prompt or an editor: this runs unattended inside a loop.
export GIT_TERMINAL_PROMPT=0 GIT_EDITOR=true

log="${1:?log file}"
name="${2:?published name}"
repo="${GITHUB_REPOSITORY:?}"
token="${GITHUB_TOKEN:?}"
url="https://x-access-token:${token}@github.com/${repo}"

source_dir="$PWD"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
# A full clone: GitHub refuses pushes from a shallow one ("shallow update not
# allowed"), and the branch is pruned to 40 small files anyway.
if ! git clone -q -b ci-logs "$url" "$work/repo" 2>/dev/null; then
  git init -q -b ci-logs "$work/repo"
fi
cd "$work/repo"
git config user.name "github-actions[bot]"
git config user.email "github-actions[bot]@users.noreply.github.com"
if [ -f "$source_dir/$log" ]; then
  cp "$source_dir/$log" "$name"
else
  echo "no log produced ($log)" >"$name"
fi
# Keep the branch small: drop logs older than the newest 40 files.
find . -maxdepth 1 -type f -name '*.log' ! -name "$name" -printf '%T@ %p
'   | sort -rn | tail -n +40 | cut -d' ' -f2- | xargs -r git rm -q --cached -- 2>/dev/null || true
git add -- "$name"
git commit -qm "ci log $name" || true
for attempt in 1 2 3 4 5; do
  if git push -q "$url" ci-logs; then
    echo "published $name to ci-logs"
    exit 0
  fi
  git pull -q --rebase "$url" ci-logs || true
  sleep "$attempt"
done
echo "could not publish $name after retries" >&2
exit 1
