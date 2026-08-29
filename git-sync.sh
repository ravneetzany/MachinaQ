#!/usr/bin/env bash
# Stage, commit, and sync MachinaQ with origin/main over SSH.
#
# Usage:
#   ./git-sync.sh "commit message"
#   ./git-sync.sh                # uses a timestamped default message
#
# What it does, in order:
#   1. Ensures origin uses the SSH remote (git@github.com:...), not HTTPS
#      (HTTPS push fails here with no stored credentials).
#   2. Stages tracked + new files, excluding local tool-config directories
#      that don't belong in the repo (.codeartsdoer/, .commandcode/, .continue/).
#   3. Commits, if there's anything staged.
#   4. Pulls --rebase from origin/main to incorporate any remote changes.
#   5. Pushes to origin/main.
#
# Safe to run with nothing to commit — it will just sync (pull --rebase)
# and skip the push if already up to date.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

REMOTE_NAME="origin"
SSH_REMOTE_URL="git@github.com:ravneetzany/MachinaQ.git"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"

EXCLUDE_PATHS=(
  ':!.codeartsdoer'
  ':!.commandcode'
  ':!.continue'
)

echo "== MachinaQ git sync =="
echo "Branch: $BRANCH"

current_url="$(git remote get-url "$REMOTE_NAME" 2>/dev/null || true)"
if [[ "$current_url" != git@github.com:* ]]; then
  echo "Switching '$REMOTE_NAME' remote to SSH ($SSH_REMOTE_URL)..."
  git remote set-url "$REMOTE_NAME" "$SSH_REMOTE_URL"
fi

echo "-- git status --"
git status --short

echo "-- staging changes (excluding local tool-config dirs) --"
git add -A -- . "${EXCLUDE_PATHS[@]}"

if git diff --cached --quiet; then
  echo "Nothing staged to commit."
else
  if [[ $# -ge 1 ]]; then
    commit_message="$1"
  else
    commit_message="Sync $(date '+%Y-%m-%d %H:%M:%S')"
  fi
  git commit -m "$commit_message"
fi

echo "-- fetching and rebasing onto $REMOTE_NAME/$BRANCH --"
git fetch "$REMOTE_NAME"
git pull --rebase "$REMOTE_NAME" "$BRANCH"

echo "-- pushing to $REMOTE_NAME/$BRANCH --"
git push "$REMOTE_NAME" "$BRANCH"

echo "== Done =="
