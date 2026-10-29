"""Produce all class, real-vs-synthetic, source and E0-E4 reports."""
import argparse,json
from dermsynreal.evaluation_pipeline import run_reporting
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--registry",required=True)
    p.add_argument("--label-audit",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--bootstrap",type=int,default=2000)
    p.add_argument("--bootstrap-seed",type=int,default=42)
    a=p.parse_args()
    report=run_reporting(a.registry,a.out,label_audit_json=a.label_audit,
                         iterations=a.bootstrap,bootstrap_seed=a.bootstrap_seed)
    print(json.dumps(report,indent=2))
if __name__=="__main__":main()
