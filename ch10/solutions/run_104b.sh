P=/tmp/claude-0/v10/bin/python
cd ../ch10/solutions
while [ ! -f run_104.done ]; do sleep 20; done
for lo in 0.05 0.1 0.2; do $P ex_10_4_qbc.py $lo 1 base > ex_10_4_5_LO${lo}_S1.out 2>&1; done
echo done > run_104b.done
