#!/bin/bash
# Start AI Workflow Studio (double-click friendly). Then open http://localhost:8080
cd "$(dirname "$0")/.." || exit 1
if systemctl is-active --quiet ai_studio 2>/dev/null; then
    IP=$(hostname -I | awk '{print $1}')
    echo "The studio is already running in the background (run-on-boot service)."
    echo "  Editor    : http://localhost:8080   or  http://$IP:8080"
    echo "  Live page : http://$IP:8080/view"
    exit 0
fi
source env/bin/activate
python3 workflow_studio/studio.py "$@"
