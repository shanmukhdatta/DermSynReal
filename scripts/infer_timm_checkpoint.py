"""Run inference on a frozen checkpoint from the four uploaded timm notebooks."""
import argparse,json
from dermsynreal.inference import run_timm_inference,MODEL_ALLOWLIST
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model-id",choices=sorted(MODEL_ALLOWLIST),required=True)
    p.add_argument("--run-config",required=True);p.add_argument("--weights",required=True)
    p.add_argument("--real-manifest",required=True);p.add_argument("--real-root",required=True)
    p.add_argument("--synthetic-manifest");p.add_argument("--synthetic-root")
    p.add_argument("--train-synthetic-manifest")
    p.add_argument("--out",required=True);p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--num-workers",type=int,default=2)
    p.add_argument("--unlock-final-test",action="store_true")
    a=p.parse_args()
    print(json.dumps(run_timm_inference(model_id=a.model_id,run_config=a.run_config,weights_file=a.weights,
    real_manifest=a.real_manifest,real_image_root=a.real_root,out_csv=a.out,
    synthetic_manifest=a.synthetic_manifest,synthetic_image_root=a.synthetic_root,
    train_synthetic_manifest=a.train_synthetic_manifest,batch_size=a.batch_size,
    num_workers=a.num_workers,allow_test=a.unlock_final_test),indent=2))
if __name__=="__main__":main()
