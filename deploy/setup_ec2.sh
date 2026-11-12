#!/bin/bash
set -euo pipefail

APP_DIR="/opt/physical-ai-system"
mkdir -p "$APP_DIR/local_data"
chown -R ubuntu:ubuntu "$APP_DIR"

# Create Python virtual environment and install dependencies
if [ ! -d "$APP_DIR/.venv" ]; then
    python3 -m venv "$APP_DIR/.venv"
fi

"$APP_DIR/.venv/bin/pip" install --upgrade pip
"$APP_DIR/.venv/bin/pip" install torch --index-url https://download.pytorch.org/whl/cpu
"$APP_DIR/.venv/bin/pip" install -r "$APP_DIR/requirements.txt"
"$APP_DIR/.venv/bin/pip" install --no-deps -e "$APP_DIR"

chmod +x "$APP_DIR/deploy/run_monthly_cron.sh"

# Install Nginx config
cp "$APP_DIR/deploy/nginx.conf" /etc/nginx/sites-available/physical-ai
ln -sf /etc/nginx/sites-available/physical-ai /etc/nginx/sites-enabled/physical-ai
rm -f /etc/nginx/sites-enabled/default
nginx -t
systemctl restart nginx

# Install systemd service
cp "$APP_DIR/deploy/physical-ai.service" /etc/systemd/system/physical-ai.service
systemctl daemon-reload
systemctl enable --now physical-ai.service
systemctl restart physical-ai.service

# Register monthly cron job (1st of every month at 03:00 JST / 18:00 UTC prev day -> run at 03:00 local)
CRON_LINE="0 3 1 * * /opt/physical-ai-system/deploy/run_monthly_cron.sh"
(crontab -u ubuntu -l 2>/dev/null | grep -v "run_monthly_cron.sh" || true; echo "$CRON_LINE") | crontab -u ubuntu -

echo "SETUP_COMPLETE"
