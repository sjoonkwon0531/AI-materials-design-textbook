P=${MLIP_PY_A:-python}
cd ../../ch10/solutions
# wait for the 10.2 chain to finish
while ! grep -q done ex_10_2_5.out 2>/dev/null; do sleep 20; done
$P ex_10_2_6.py > ex_10_2_6.out 2>&1
$P ex_10_4_qbc.py 0.05 0 base > ex_10_4_5_LO0.05_S0.out 2>&1
$P ex_10_4_qbc.py 0.1 0 base > ex_10_4_5_LO0.1_S0.out 2>&1
$P ex_10_4_qbc.py 0.2 0 base > ex_10_4_5_LO0.2_S0.out 2>&1
$P ex_10_4_5c.py > ex_10_4_5c.out 2>&1
$P ex_10_4_qbc.py 0.05 0 mixed > ex_10_4_6_run.out 2>&1
$P ex_10_4_4.py big > ex_10_4_4_big.out 2>&1
echo alldone > run_104.done
