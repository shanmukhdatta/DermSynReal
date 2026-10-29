# Author-controlled morphology prompt collections

Six folders: `Acne`, `Fungal_Infection`, `Vitiligo`, `Hyperpigmentation`, `Skin_Cancer`, `Normal_Skin`.

Each has **300 authored prompts in JSON/CSV/TXT**, with explicit `prompt_id`, `prompt_group_id`, `positive_prompt`, `negative_prompt`, `split` and `seed_base`. Each class has 60 prompt families, with group-disjoint 210 train / 45 val / 45 test. Keep **positive and negative prompt text exactly as authored**; do not rebuild or truncate it silently. The generation notebooks now load these files through `src/dermsynreal/kaggle_runs.py`. Pilot five **train** prompts per class; then expand to all train prompts only after tokenizer, checkpoint and morphology review.

The `Skin_Cancer` prompts include AK-like **precancerous** appearances, not just invasive cancers; fungal prompts include candidiasis-like **unconfirmed** patterns. Appearance text cannot establish real diagnosis. Generated images are unreviewed until a qualified review sets `accepted=True`.

Original raw Kaggle v2 real image source: `shanmukhadattaboda/facial-skin-conditions-real-image-collection`. Do not use raw clinical photographs as generator training unless first approved and frozen into a train-only split. Synthetic generation uses no real images as prompts by default.
