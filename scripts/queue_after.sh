#!/usr/bin/env sh
# Wait for a process to exit, then run a command. For chaining CPU training jobs on a
# box without a scheduler.
# Usage: nohup scripts/queue_after.sh <pid> <command...> > logs/x.log 2>&1 &
set -u
PID="$1"
shift
while kill -0 "$PID" 2>/dev/null; do
    sleep 60
done
echo "pid $PID exited at $(date); starting: $*"
exec "$@"
