import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from statistical_analysis import run_phase8_analysis

if __name__ == '__main__':
    run_phase8_analysis(exp_id='C-Arch-05-MS', n_boot=2000)
