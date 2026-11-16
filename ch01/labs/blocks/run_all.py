"""Run the Ch.1 code blocks in the order they appear in the book.

The blocks are consecutive cells from the chapter text: later blocks reuse
variables defined by earlier ones, so they share one namespace here, as in a
notebook. Run from this folder:  python run_all.py
b00 is an interactive-prompt transcript and b19 is an illustrative snippet
(energy_model is not defined in the text), so both are shown but not run.
"""
import time, pathlib
HERE = pathlib.Path(__file__).resolve().parent
ORDER = ['b01', 'b02', 'b03', 'b04', 'b05', 'c143', 'b06', 'b07', 'b08', 'b09', 'b10',
         'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b17', 'b18', 'b20', 'b21']
ns = {'__name__': '__main__'}
for name in ORDER:
    path = HERE / f'{name}.py'
    print(f'\n===== {name}.py =====', flush=True)
    t = time.time()
    exec(compile(path.read_text(), str(path), 'exec'), ns)
    print(f'[{name} done in {time.time() - t:.1f} s]', flush=True)
