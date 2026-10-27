#!/bin/bash
# =============================================================================
# DETECTION 5 - RUN YOUR AI APP AUTOMATICALLY WHEN THE JETSON STARTS
# =============================================================================
# Real products don't need someone to type "python3 ...". This installs the web
# dashboard (d4) as a systemd service: it starts on boot and restarts if it crashes.
#
#   bash install_service.sh                  # install + start
#   bash install_service.sh remove           # stop + uninstall
#
# Useful commands afterwards:
#   systemctl status ai_camera               # is it running?
#   journalctl -u ai_camera -f               # live log (Ctrl+C to exit)
#   sudo systemctl restart ai_camera         # restart after changing the code
# =============================================================================
set -e
NAME=ai_camera
DIR="$(cd "$(dirname "$0")" && pwd)"            # this folder (object_detection)
PY="$(cd "$DIR/.." && pwd)/env/bin/python3"     # the workshop virtual environment
MODEL=yolo26n.pt
[ -f "$DIR/yolo26n.engine" ] && MODEL=yolo26n.engine   # use TensorRT if you built it

if [ "$1" == "remove" ]; then
    sudo systemctl disable --now $NAME 2>/dev/null || true
    sudo rm -f /etc/systemd/system/$NAME.service
    sudo systemctl daemon-reload
    echo "Removed $NAME service."
    exit 0
fi

[ -x "$PY" ] || { echo "Can't find $PY - is the venv at ~/ai_workshop/env ?"; exit 1; }

sudo tee /etc/systemd/system/$NAME.service > /dev/null <<UNIT
[Unit]
Description=Jetson AI Camera (YOLO web dashboard)
After=network-online.target
Wants=network-online.target

[Service]
User=$USER
WorkingDirectory=$DIR
ExecStart=$PY $DIR/d4_web_dashboard.py --model $MODEL
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT

sudo systemctl daemon-reload
sudo systemctl enable --now $NAME
sleep 3
systemctl --no-pager status $NAME | head -5
echo
echo "Installed! It now starts every time the Jetson boots."
echo "Open:  http://$(hostname -I | awk '{print $1}'):5000"
