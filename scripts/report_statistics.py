"""Report real-test descriptive means, sample SD and paired differences across seeds."""
import argparse
import json
from dermsynreal.statistical_analysis import summarize_across_seeds

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--results-dir",required=True)
    parser.add_argument("--out")
    args=parser.parse_args()
    print(json.dumps(summarize_across_seeds(args.results_dir,args.out),indent=2))

if __name__=="__main__":
    main()
