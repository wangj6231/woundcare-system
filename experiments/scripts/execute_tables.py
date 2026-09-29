import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from generate_tables import run_phase9_table_generation

if __name__ == '__main__':
    run_phase9_table_generation()
