# Data directory (metadata only)

Never commit raw patient photographs, unrestricted scraped images, large synthetic corpora, model weights, personal details, or Kaggle credentials.

Store the actual image bytes in properly permissioned Kaggle Datasets or other authorized private storage. Keep lightweight **non-identifying** manifests and split hashes in version control as policy permits.

Suggested external mounts:
- `/kaggle/input/dermsynreal-real-images/`
- `/kaggle/input/dermsynreal-frozen-splits/`
- `/kaggle/input/dermsynreal-synthetic-v1/`

Subdirectories:
- `manifests/`: versioned metadata schemas/templates
- `splits/`: approved immutable partition manifests (if privacy-safe)
- `synthetic/`: only metadata/examples permitted by license
