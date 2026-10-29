"""Combine reviewed synthetic generation outputs after reattaching saved Kaggle run folders."""
import argparse,json
from dermsynreal.combine_synthetic import combine_approved
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--manifests",nargs="+",required=True,
                   help="Reviewed generation_manifest.csv files, one per run")
    p.add_argument("--image-roots",nargs="+",required=True,
                   help="Respective mounted original generation output folders")
    p.add_argument("--out",required=True)
    p.add_argument("--generator-mode",help="Require a particular base/TI/LoRA mode")
    p.add_argument("--partition",choices=["train","val","test"])
    p.add_argument("--require-six-classes",action="store_true")
    p.add_argument("--allow-mixed-generators",action="store_true",
                   help="Preprocessing only: preserve base/LoRA modes in one reviewed pool")
    a=p.parse_args()
    print(json.dumps(combine_approved(a.manifests,a.image_roots,a.out,
                  generator_mode=a.generator_mode,
                  prompt_partition=a.partition,require_six_classes=a.require_six_classes,
                  allow_mixed_generators=a.allow_mixed_generators),indent=2))
if __name__=="__main__":
    main()
