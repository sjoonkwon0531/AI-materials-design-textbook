#!/bin/bash
# Runs every verification script for the Ch.2 (part b) solutions; outputs go to outputs/<script>.txt
cd "$(dirname "$0")"
for f in ex2_3_A ex2_3_BC ex2_4_4 ex2_4_5 ex2_4_6 ex2_4_7 ex2_4_8 ex2_4_9 ex2_5_1 ex2_5_2 ex2_5_3 ex2_5_4 ex2_5_5 ex2_5_6 \
         ex2_6_1_2 ex2_6_3 ex2_6_4 ex2_6_5 ex2_6_5_ch ex2_6_6 ex2_7_1 ex2_7_2 ex2_7_3 ex2_7_4 ex2_7_5; do
  start=$(date +%s)
  python3 $f.py > outputs/$f.txt 2>&1
  echo "$f exit=$? $(( $(date +%s) - start ))s" >> outputs/_status.txt
done
echo DONE >> outputs/_status.txt
