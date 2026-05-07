import sys
sys.path.insert(0, '.')
import os

# When running with -m src.experiments.train_unichiflow from unidexflow dir:
# __file__ should be src/experiments/train_unichiflow.py
# _script_dir should be G:\AMark\Code\unidexflow\src\experiments
# config_path should be G:\AMark\Code\unidexflow\src\experiments\..\..\configs
# which normalizes to G:\AMark\Code\unidexflow\configs

_script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(_script_dir, '..', '..', 'configs')

with open('debug_output.txt', 'w') as f:
    f.write(f'__file__: {__file__}\n')
    f.write(f'cwd: {os.getcwd()}\n')
    f.write(f'_script_dir: {_script_dir}\n')
    f.write(f'config_path: {config_path}\n')
    f.write(f'exists: {os.path.exists(config_path)}\n')
    f.write(f'abs config_path: {os.path.abspath(config_path)}\n')
    f.write(f'abs exists: {os.path.exists(os.path.abspath(config_path))}\n')

print('Debug output written to debug_output.txt')
print(f'config_path: {config_path}')
print(f'exists: {os.path.exists(config_path)}')