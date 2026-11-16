import os, sys, importlib
LABS = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "labs"))
sys.path.insert(0, LABS)
def lab(name):
    return importlib.import_module(name)
