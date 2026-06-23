import os
import sys

# Add the project root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.phase_diagram import run_grid

def main():
    print("============================================================")
    print("TASK 8: T-x PHASE DIAGRAM CONSTRUCTION")
    print("============================================================")
    run_grid()

if __name__ == '__main__':
    main()
