import os
import sys
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.evaluate_fresh_checkpoints import main as evaluate_checkpoints


if __name__ == "__main__":
    # Run the checkpoint evaluation
    evaluate_checkpoints() 