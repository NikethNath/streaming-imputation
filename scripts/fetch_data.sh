#!/usr/bin/env bash
# Fetch the AQI-36 benchmark (36 Beijing PM2.5 stations, hourly, 12 months).
#
# This is the archive released with ST-MVL (Yi et al., IJCAI 2016) and used by
# both CSDI and PriSTI. We pull it from the PriSTI repository, which ships it
# alongside the station coordinates we need for the neighbour features.
set -euo pipefail
cd "$(dirname "$0")/.."
DEST="data/pm25/SampleData"
BASE="https://raw.githubusercontent.com/LMZZML/PriSTI/main/data/pm25"
mkdir -p "$DEST"
for f in pm25_ground.txt pm25_missing.txt pm25_latlng.txt; do
  if [ -s "$DEST/$f" ]; then
    echo "have $f"
  else
    echo "fetching $f"
    curl -fsSL "$BASE/SampleData/$f" -o "$DEST/$f"
  fi
done
[ -s data/pm25/pm25_meanstd.pk ] || curl -fsSL "$BASE/pm25_meanstd.pk" -o data/pm25/pm25_meanstd.pk
echo
echo "ground truth: $(wc -l < "$DEST/pm25_ground.txt") rows"
echo "stations:     $(head -1 "$DEST/pm25_ground.txt" | tr ',' '\n' | tail -n +2 | wc -l)"
