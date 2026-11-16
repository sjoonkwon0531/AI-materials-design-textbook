"""Common helpers for the Section 10.3 benchmark labs (four universal MLIPs, two interpreters).

Usage inside a lab script:
    from importlib import import_module
    C = import_module('103_common'); C.reexec_for(model)   # switch to the right venv if needed
    calc, net, info = C.make_calc(model)
"""
import os, sys, time, json, warnings

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), 'data')
PY = {'mace': '/tmp/claude-0/v10/bin/python', 'chgnet': '/tmp/claude-0/v10/bin/python',
      'sevennet': '/tmp/claude-0/v10b/bin/python', 'mattersim': '/tmp/claude-0/v10b/bin/python'}
LABEL = {'mace': 'MACE-MP-0 (small)', 'chgnet': 'CHGNet 0.3.0',
         'sevennet': 'SevenNet-0', 'mattersim': 'MatterSim-v1.0.0-1M'}
MODELS = ['mace', 'chgnet', 'sevennet', 'mattersim']


def reexec_for(model):
    """Re-launch this script under the interpreter that has `model` installed."""
    want = PY[model]
    if os.path.realpath(sys.prefix) != os.path.realpath(os.path.dirname(os.path.dirname(want))):
        os.execv(want, [want] + sys.argv)


def make_calc(model):
    warnings.filterwarnings('ignore')
    import torch
    torch.set_num_threads(1)
    t0 = time.process_time()
    if model == 'mace':
        from mace.calculators import MACECalculator
        calc = MACECalculator(model_paths=os.path.join(DATA, 'mace_mp0_small.model'),
                              device='cpu', default_dtype='float64')
        net = calc.models[0]
    elif model == 'chgnet':
        from chgnet.model import CHGNet, CHGNetCalculator
        net = CHGNet.load(verbose=False, use_device='cpu')
        calc = CHGNetCalculator(model=net, use_device='cpu')
    elif model == 'sevennet':
        from sevenn.calculator import SevenNetCalculator
        calc = SevenNetCalculator('7net-0', device='cpu')
        net = calc.model
    elif model == 'mattersim':
        from loguru import logger
        logger.remove()
        from mattersim.forcefield import MatterSimCalculator
        calc = MatterSimCalculator(device='cpu')
        net = calc.potential.model
    else:
        raise ValueError(model)
    info = {'label': LABEL[model], 'n_params': int(sum(p.numel() for p in net.parameters())),
            'dtype': str(next(net.parameters()).dtype).replace('torch.', ''),
            'load_cpu_s': round(time.process_time() - t0, 2)}
    return calc, net, info


def save(model, part, payload):
    """Merge `payload` under key `part` into data/103_<model>.json."""
    path = os.path.join(DATA, f'103_{model}.json')
    d = json.load(open(path)) if os.path.exists(path) else {}
    d[part] = payload
    json.dump(d, open(path, 'w'), indent=1)
    return path
