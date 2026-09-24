#!/bin/bash
# Subprocess-isolated OpenML arm: hard per-dataset timeout, immune to C-level hangs.
cd "$(dirname "$0")"
for did in 8 25 163 171 224 346 481 553 1039 1084 1101 1107 1122 1126 1128 1130 1245 1412 1432 1434 1464 1465 1466 1471 1498 1506 1523 4134 4531 4540 4541 40474 40477 40966 41430 42878 42893 42895 42900 42901 42972 43008 43414 43428 43657 43658 43726 43439 43284; do
  timeout -k 20 420 python3 run_openml.py "$did"
  rc=$?
  [ $rc -eq 124 ] || [ $rc -eq 137 ] && echo "FAIL $did hard-timeout" 
done
echo "ALL_OPENML_PASS_COMPLETE"
