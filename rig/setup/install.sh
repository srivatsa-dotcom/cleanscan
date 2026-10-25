#!/usr/bin/env bash
# One-time setup of a Raspberry Pi Zero W / Zero 2 W as a CleanScan LiDAR rig.
# Raspberry Pi OS Bookworm (Lite is enough). Run from the rig/ folder:
#     sudo ./setup/install.sh ld06            # or: ld19, tfluna, tfmini
# Then reboot. The Pi creates the Wi-Fi network "CleanScanRig" (password below) and
# streams the LiDAR to the app on 10.42.0.1:5600.
set -euo pipefail

SENSOR="${1:-ld06}"
SSID="${CLEANSCAN_SSID:-CleanScanRig}"
PASS="${CLEANSCAN_PASS:-cleanscan123}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ "$(id -u)" -ne 0 ]; then echo "run with sudo"; exit 1; fi

echo "== packages"
apt-get update
apt-get install -y python3-venv python3-serial network-manager

echo "== Python package"
python3 -m venv --system-site-packages /opt/cleanscan-rig
/opt/cleanscan-rig/bin/pip install --no-deps "$HERE"

echo "== serial port: hardware UART on GPIO14/15, no login console"
raspi-config nonint do_serial_cons 1   # disable the serial login console
raspi-config nonint do_serial_hw 0     # enable the UART
CONFIG=/boot/firmware/config.txt; [ -f "$CONFIG" ] || CONFIG=/boot/config.txt
# give the stable PL011 UART to the GPIO pins (Bluetooth is not needed)
grep -q '^dtoverlay=disable-bt' "$CONFIG" || echo 'dtoverlay=disable-bt' >> "$CONFIG"
systemctl disable --now hciuart 2>/dev/null || true

echo "== service"
cat > /etc/systemd/system/cleanscan-rig.service <<EOF
[Unit]
Description=CleanScan LiDAR rig bridge
After=network-online.target

[Service]
ExecStart=/opt/cleanscan-rig/bin/cleanscan-rig --sensor $SENSOR --serial /dev/serial0
Restart=always
RestartSec=2

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable cleanscan-rig

echo "== Wi-Fi hotspot $SSID (the Pi will be 10.42.0.1; starts after the reboot)"
nmcli connection delete cleanscan-rig 2>/dev/null || true
nmcli connection add type wifi ifname wlan0 con-name cleanscan-rig autoconnect yes save yes ssid "$SSID" \
    802-11-wireless.mode ap 802-11-wireless.band bg ipv4.method shared \
    wifi-sec.key-mgmt wpa-psk wifi-sec.psk "$PASS"
nmcli connection modify cleanscan-rig connection.autoconnect-priority 100

echo
echo "Done. Reboot now:  sudo reboot"
echo "Afterwards join the Wi-Fi '$SSID' (password '$PASS') from the iPhone, open CleanScan,"
echo "and the status line shows the rig. Check it from a laptop on the same Wi-Fi with:"
echo "    cleanscan-rig-monitor --host 10.42.0.1"
echo "Note: while the hotspot is on, the Pi has no internet; SSH to it at 10.42.0.1."
