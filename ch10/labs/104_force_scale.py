"""RMS force component of each Section 10.4 data set (scale for reading the force RMSEs)."""
import numpy as np
for n in ['104_test', '104_transfer_1800K', '104_transfer_strain', '104_transfer_vacancy', '104_rattle', '104_pool']:
    d = np.load('../data/' + n + '.npz'); F = d['F']
    print(n, len(F), 'RMS force component %.0f meV/A' % (1000 * np.sqrt((F ** 2).mean())))
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
C = import_module('104_common')
print('student SchNet parameters:', sum(p.numel() for p in C.new_student(0, 0.0).parameters()))
d = np.load('../data/104_data_fig.npz')
print('force-ACF minimum at lag (fs):', {T: int(2 * np.argmin(d[f'acf{T}'])) for T in (300, 900, 1500)})
