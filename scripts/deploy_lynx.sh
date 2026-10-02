#!/usr/bin/env sh
# Push the committed tree (HEAD) to the training box, which has no git.
# Usage: scripts/deploy_lynx.sh [user@host]   (default marwin@100.96.206.25)
set -eu
HOST="${1:-marwin@100.96.206.25}"
git archive --format=tar HEAD | ssh "$HOST" 'mkdir -p ~/varroa-vision && tar -x -C ~/varroa-vision && echo deployed $(date)'
