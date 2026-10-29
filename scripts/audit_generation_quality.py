"""Summarize manual morphology/near-copy review without making clinical claims."""
import argparse,json
from dermsynreal.quality_audit import summarize_quality
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--reviews",required=True)
    p.add_argument("--out",required=True)
    a=p.parse_args()
    print(json.dumps(summarize_quality(a.reviews,a.out),indent=2))
if __name__=="__main__":main()
