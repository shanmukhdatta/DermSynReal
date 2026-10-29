"""Optional confidence calibration report, independently by test domain."""
import argparse,json
from pathlib import Path
from dermsynreal.calibration import calibration_by_domain
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--predictions",required=True);p.add_argument("--out",required=True)
    p.add_argument("--bins",type=int,default=10)
    a=p.parse_args()
    data=calibration_by_domain(a.predictions,a.bins)
    o=Path(a.out);o.parent.mkdir(parents=True,exist_ok=True)
    o.write_text(json.dumps(data,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:{name:v[name] for name in ("n","ece","multiclass_brier","nll")} for k,v in data.items()},indent=2))
if __name__=="__main__":main()
