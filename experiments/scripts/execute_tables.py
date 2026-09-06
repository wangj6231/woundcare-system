import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from generate_tables import run_phase9_table_generation

run_phase9_table_generation()
