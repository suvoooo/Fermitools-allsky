#!/bin/bash
# Downloads 16 years of Fermi-LAT weekly spacecraft files.
# similar structure to the photon bash file
#
# Before running:
#   1. Run:  python3 generate_photon_urls.py
#      This produces spacecraft_urls.txt (see script output for file count).
#   2. Then: nohup bash this_file > logs/nohup_spacecraft.log 2>&1 &

BASE_DIR="/d6/CAC/sbhattacharyya/Documents/data/fermi16-yrs"
SC_DIR="${BASE_DIR}/spacecraft"
LOG_DIR="${BASE_DIR}/logs"

WGET_OPTS="-c --tries=0 --waitretry=30 --retry-connrefused --timeout=60"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SC_URL_FILE="${SCRIPT_DIR}/spacecraft_urls.txt"

if [ ! -f "$SC_URL_FILE" ]; then
    echo "ERROR: spacecraft_urls.txt not found at ${SC_URL_FILE}"
    echo "Run:  python3 generate_photon_urls.py  to generate it."
    exit 1
fi

TOTAL=$(wc -l < "$SC_URL_FILE")
echo "Starting download of ${TOTAL} weekly spacecraft files..."
echo "Log: ${LOG_DIR}/spacecraft_download.log"

wget $WGET_OPTS \
     -P "$SC_DIR" \
     -i "$SC_URL_FILE" \
     -a "${LOG_DIR}/spacecraft_download.log"

echo "Spacecraft file download complete."
