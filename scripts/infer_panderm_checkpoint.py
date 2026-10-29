"""Run inference using official PanDerm encoder plus saved linear probe."""
import argparse,json
from dermsynreal.panderm_inference import run_panderm_inference
def main():
    p=argparse.ArgumentParser()
    for key in ("official-repo","official-weights","saved-probe","model-details",
                "real-manifest","real-root","out"):
        p.add_argument("--"+key,required=True)
    p.add_argument("--synthetic-manifest");p.add_argument("--synthetic-root")
    p.add_argument("--train-synthetic-manifest")
    p.add_argument("--batch-size",type=int,default=4)
    p.add_argument("--num-workers",type=int,default=2)
    p.add_argument("--unlock-final-test",action="store_true")
    a=p.parse_args()
    print(json.dumps(run_panderm_inference(official_repo=a.official_repo,
    official_weights=a.official_weights,saved_probe=a.saved_probe,
    model_details=a.model_details,real_manifest=a.real_manifest,real_image_root=a.real_root,
    out_csv=a.out,synthetic_manifest=a.synthetic_manifest,synthetic_image_root=a.synthetic_root,
    train_synthetic_manifest=a.train_synthetic_manifest,batch_size=a.batch_size,
    num_workers=a.num_workers,allow_test=a.unlock_final_test),indent=2))
if __name__=="__main__":main()
