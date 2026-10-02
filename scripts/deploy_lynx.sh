#!/usr/bin/env sh
# Push the committed tree (HEAD) to the training box, which has no git.
# Usage: scripts/deploy_lynx.sh [ssh-target]
# Default: $VARROA_TRAIN_HOST, else the ssh config alias "lynx" (keep the address in
# ~/.ssh/config, not in the repo).
set -eu
HOST="${1:-${VARROA_TRAIN_HOST:-lynx}}"
git archive --format=tar HEAD | ssh "$HOST" 'mkdir -p ~/varroa-vision && tar -x -C ~/varroa-vision && echo deployed $(date)'
