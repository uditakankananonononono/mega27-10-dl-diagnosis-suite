#!/bin/bash
# Subprocess-isolated GEO pass-2 driver: hard per-GSE timeout, hang-proof.
cd "$(dirname "$0")"
for gse in $@; do
  timeout -k 20 300 python3 -u run_geo2.py "$gse"
  rc=$?
  if [ $rc -eq 124 ] || [ $rc -eq 137 ]; then
    echo "FAIL $gse hard-timeout"
    echo "$gse" >> logs/geo_hung.txt
  fi
done
echo "GEO2_ALL_COMPLETE"
