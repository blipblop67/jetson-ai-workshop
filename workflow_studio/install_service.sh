#!/bin/bash
# =============================================================================
# AI WORKFLOW STUDIO - START AUTOMATICALLY WHEN THE JETSON BOOTS (trainer, once)
# =============================================================================
# After this, the studio runs in the background all the time, and the project you
# DEPLOY starts again by itself after a reboot or power cut - like a real product.
#
#   bash workflow_studio/install_service.sh            # install + start
#   bash workflow_studio/install_service.sh remove     # stop + uninstall
#
#   systemctl status ai_studio        # is it running?
#   journalctl -u ai_studio -f        # live log (Ctrl+C to exit)
# =============================================================================
set -e
NAME=ai_studio
DIR="$(cd "$(dirname "$0")" && pwd)"            # workflow_studio
PY="$(cd "$DIR/.." && pwd)/env/bin/python3"     # the workshop virtual environment

if [ "$1" == "remove" ]; then
    sudo systemctl disable --now $NAME 2>/dev/null || true
    sudo rm -f /etc/systemd/system/$NAME.service
    sudo systemctl daemon-reload
    echo "Removed. Start the studio by hand again with: bash workflow_studio/start_studio.sh"
    exit 0
fi

[ -x "$PY" ] || { echo "Can't find $PY - is the venv at ~/ai_workshop/env ?"; exit 1; }
if ss -ltn 2>/dev/null | grep -q ':8080 '; then
    echo "Port 8080 is in use - stop the studio you started by hand (Ctrl+C in its terminal) and run this again."
    exit 1
fi

sudo tee /etc/systemd/system/$NAME.service > /dev/null <<UNIT
[Unit]
Description=AI Workflow Studio (Jetson)
After=network-online.target
Wants=network-online.target

[Service]
User=$USER
WorkingDirectory=$DIR
ExecStart=$PY $DIR/studio.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT

sudo systemctl daemon-reload
sudo systemctl enable --now $NAME
sleep 3
systemctl --no-pager status $NAME | head -5
IP=$(hostname -I | awk '{print $1}')
echo
echo "Installed! The studio now starts every time the Jetson boots."
echo "  Editor    : http://$IP:8080"
echo "  Live page : http://$IP:8080/view"
