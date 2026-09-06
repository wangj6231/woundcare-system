#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
mkdir -p label_studio_data exports
sudo chown -R 1001:1001 label_studio_data exports
echo "Prepared writable directories for Label Studio UID 1001."
