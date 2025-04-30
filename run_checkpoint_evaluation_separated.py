import os
import sys
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def parse_args():
    parser = argparse.ArgumentParser(description="Run checkpoint evaluation pipeline")
    parser.add_argument("--generate_only", action="store_true", 
                        help="Only generate datasets, don't evaluate")
    parser.add_argument("--evaluate_only", action="store_true",
                        help="Only evaluate existing datasets")
    parser.add_argument("--dataset_results", type=str, default=None,
                        help="Path to dataset generation results JSON file (for evaluate_only)")
    
    # Pass through all other arguments
    parser.add_argument('args', nargs=argparse.REMAINDER)
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    if args.evaluate_only:
        # Run evaluation only
        from scripts.evaluate_fresh_datasets import main as evaluate_datasets
        
        # Prepare sys.argv for the evaluate script
        sys.argv = [sys.argv[0]] + args.args
        if args.dataset_results:
            sys.argv.extend(["--dataset_results", args.dataset_results])
        
        evaluate_datasets()
    elif args.generate_only:
        # Run generation only
        from scripts.generate_fresh_datasets import main as generate_datasets
        
        # Prepare sys.argv for the generate script
        sys.argv = [sys.argv[0]] + args.args
        
        generate_datasets()
    else:
        # Run both
        from scripts.generate_fresh_datasets import main as generate_datasets
        from scripts.evaluate_fresh_datasets import main as evaluate_datasets
        
        # Prepare sys.argv for the generate script
        sys.argv = [sys.argv[0]] + args.args
        
        # Generate datasets
        generate_datasets()
        
        # Evaluate datasets
        evaluate_datasets() 