# Publishing checklist

1. Publish only this standalone directory, NOT the parent research repository.
   Do not upload paper drafts, reviews, SSH settings, credentials, private logs,
   datasets, full training checkpoints, or EviPose experiments.
2. Inspect `reports/` and each weight manifest. Report measured AP honestly;
   output parity on a few images does not establish full-dataset AP.
3. Create a new GitHub repository. Commit the files below explicitly (not the
   parent workspace). `git init` belongs INSIDE this release directory only.
4. Attach the two `.pt` model files to a tagged GitHub Release, rather than to
   Git history. Add the actual download links to README after the release exists.
5. Optionally attach full `predictions.json` as Release assets, alongside a
   file checksum. Keep the much smaller `metrics.json` and parity reports in Git.
6. Include `assets/` so both paper figures render in the README. Preserve their
   values and distinguish the paper's TensorRT benchmark from release evaluation.
7. Retain license/attribution. Confirm the authors approve publication and the
   weights' distribution terms. The public repository is https://github.com/eplius/VFR.

Example commands for an initial local publication (not needed after web upload):

```bash
git init
git add README.md PUBLISHING.md LICENSE NOTICE requirements.txt .gitignore
git add runtime.py infer.py evaluate.py evaluate_coco.sh weights/*.md weights/*.json reports/ assets/
git commit -m "Release VFR-S/M COCO inference and evaluation"
git branch -M main
git remote add origin https://github.com/eplius/VFR.git
git push -u origin main
```

GitHub Release assets: `weights/vfr_s_coco.pt`, `weights/vfr_m_coco.pt`.
The README reports S=67.6 and M=70.6 AP; metric JSON files retain full precision.
Add the approved paper URL, citation, and actual model-download links before
publication. Do not invent publication status or a repository URL.

Suggested GitHub description:

> Visibility-aware Fine-grained Refinement for lightweight end-to-end multi-person
> pose estimation. VFR-S/M pretrained models and reproducible COCO evaluation.

Do not claim full training reproducibility or re-measured TensorRT latency from this package.
