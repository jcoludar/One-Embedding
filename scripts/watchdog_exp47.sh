#!/bin/bash
# Watchdog for Exp 47 codec sweep
# Checks every 10 minutes: process alive, log growing, disk not full, no errors
# Run: nohup bash scripts/watchdog_exp47.sh &

LOG="results/exp47_full_sweep.log"
PID=79482
CHECK_INTERVAL=600  # 10 minutes
PREV_SIZE=0

echo "$(date): Watchdog started for PID $PID"
echo "  Log: $LOG"
echo "  Check interval: ${CHECK_INTERVAL}s"

while true; do
    sleep $CHECK_INTERVAL

    # Is the process still running?
    if ! kill -0 $PID 2>/dev/null; then
        echo "$(date): DONE — Process $PID has exited"
        echo "  Final log size: $(wc -c < $LOG 2>/dev/null) bytes"
        echo "  Last 5 lines:"
        tail -5 "$LOG" 2>/dev/null
        break
    fi

    # Is the log growing?
    CURR_SIZE=$(wc -c < "$LOG" 2>/dev/null || echo 0)
    if [ "$CURR_SIZE" = "$PREV_SIZE" ]; then
        echo "$(date): WARNING — Log not growing (stuck at $CURR_SIZE bytes)"
        echo "  CPU: $(ps -p $PID -o %cpu= 2>/dev/null || echo 'N/A')%"
        echo "  MEM: $(ps -p $PID -o %mem= 2>/dev/null || echo 'N/A')%"
    else
        DELTA=$((CURR_SIZE - PREV_SIZE))
        echo "$(date): OK — Log grew +${DELTA} bytes (total: ${CURR_SIZE})"
    fi
    PREV_SIZE=$CURR_SIZE

    # Check for errors in recent log output
    RECENT_ERRORS=$(tail -50 "$LOG" 2>/dev/null | grep -c -i "error\|traceback\|exception\|killed")
    if [ "$RECENT_ERRORS" -gt 0 ]; then
        echo "$(date): WARNING — $RECENT_ERRORS error-like lines in last 50 lines"
        tail -5 "$LOG" 2>/dev/null | head -3
    fi

    # Check disk space
    DISK_FREE=$(df -g "$HOME" | tail -1 | awk '{print $4}')
    if [ "$DISK_FREE" -lt 10 ]; then
        echo "$(date): CRITICAL — Only ${DISK_FREE}GB free disk space!"
    fi

    # Show progress
    CONFIGS_DONE=$(grep -c "SS3=" "$LOG" 2>/dev/null || echo 0)
    echo "  Progress: ~$CONFIGS_DONE config results so far"

done

echo "$(date): Watchdog exiting"
