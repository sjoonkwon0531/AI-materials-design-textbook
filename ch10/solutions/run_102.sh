P=${MLIP_PY_A:-python}
cd ../../ch10/solutions
$P ex_10_2_3.py 0 10 > ex_10_2_3a.out 2>&1
$P ex_10_2_3.py 100 1000 > ex_10_2_3b.out 2>&1
for s in 0 1 2; do $P ex_10_2_5.py inv 52 $s >> ex_10_2_5.out 2>&1; done
for s in 1 2; do $P ex_10_2_5.py inv 32 $s >> ex_10_2_5.out 2>&1; $P ex_10_2_5.py eq 32 $s >> ex_10_2_5.out 2>&1; done
echo done >> ex_10_2_5.out
