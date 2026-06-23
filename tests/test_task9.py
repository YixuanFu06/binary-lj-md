import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.nnff_compare import single_point_comparison, pes_comparison, short_md, benchmark_performance

def main():
    print("============================================================")
    print("TASK 9: M3GNet Neural Network Force Field Comparison")
    print("============================================================")
    
    single_point_comparison()
    pes_comparison()
    short_md()
    benchmark_performance()

if __name__ == '__main__':
    main()
