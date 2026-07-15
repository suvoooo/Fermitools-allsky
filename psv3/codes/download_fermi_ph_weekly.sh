#!/bin/bash
# Downloads 16 years of Fermi-LAT weekly photon files .
# wget -c will resume/skip files
# already downloaded 
#
# Before running:
#   1. Go to https://fermi.gsfc.nasa.gov/ssc/data/analysis/scitools/LAT_weekly_allsky.html
#   2. Follow instructions to generate the wget URL list for weekly photon files
#   3. Save that list as photon_urls.txt in the same directory as this script
#   4. Save the spacecraft file URL as spacecraft_url.txt (single line)
#
# Then run this bash

BASE_DIR="/d6/CAC/sbhattacharyya/Documents/data/fermi16-yrs"
PHOTON_DIR="${BASE_DIR}/photon"
LOG_DIR="${BASE_DIR}/logs"

WGET_OPTS="-c --tries=0 --waitretry=30 --retry-connrefused --timeout=60"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Photon files ──────────────────────────────────────────────────────────────
PHOTON_URL_FILE="${SCRIPT_DIR}/photon_urls.txt"

if [ ! -f "$PHOTON_URL_FILE" ]; then
    echo "ERROR: photon_urls.txt not found at ${PHOTON_URL_FILE}"
    echo "Please generate this file from the FSSC website and place it alongside this script."
    exit 1
fi

TOTAL=$(wc -l < "$PHOTON_URL_FILE")
echo "Starting download of ${TOTAL} weekly photon files..."
echo "Log: ${LOG_DIR}/photon_download.log"

wget $WGET_OPTS \
     -P "$PHOTON_DIR" \
     -i "$PHOTON_URL_FILE" \
     -a "${LOG_DIR}/photon_download.log"

echo "Photon file download complete."

