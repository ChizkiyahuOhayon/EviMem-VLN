# Models, datasets, and licenses

## Automatically downloadable assets

The model script pins immutable Hugging Face revisions:

| Local asset | Source | Revision |
|---|---|---|
| GA-VLN checkpoint | `jahhao/gavln_official` | `b97f18b48ffc53ab4145d7bbdb517945f3464162` |
| SigLIP2 So400m/14 384 | `google/siglip2-so400m-patch14-384` | `e8e487298228002f3d8a82e0cd5c8ea9c567f57f` |
| VGGT-1B | `facebook/VGGT-1B` | `860abec7937da0a4c03c41d3c269c366e82abdf9` |

The downloader omits duplicate or training-only files and writes a local
SHA-256 manifest after each completed model.

R2R-CE and RxR-CE episode archives are downloaded from the official VLN-CE
Google Drive IDs and normalized into GA-VLN-compatible directories.

## Matterport3D

MP3D scenes cannot be redistributed or downloaded anonymously by this project.
First accept the Matterport3D Terms of Use and obtain the official
`download_mp.py`. Then run:

```bash
bash scripts/download_mp3d.sh \
  --asset-root /nas/evimem-assets \
  --download-script /secure/path/download_mp.py \
  --i-accept-mp3d-terms
```

The script requires explicit acknowledgment and verifies all 90 Habitat `.glb`
scenes. It does not bundle credentials or bypass the official process.

## NAS layout

```text
/nas/evimem-assets/
├── checkpoints/gavln_official/
├── model/siglip-so400m-patch14-384/
├── model/VGGT-1B/
└── vln_data/
    ├── datasets/r2r/
    ├── datasets/RxR_VLNCE_v0/
    └── scene_datasets/mp3d/
```

Treat this directory as immutable canonical storage. Stage it to local NVMe for
each experiment; do not run Habitat directly against a slow network filesystem.

## License boundary

- VLN-CE code is MIT licensed. Its MP3D-derived R2R/RxR task data is distributed
  under the Matterport3D Terms and CC BY-NC-SA 3.0 US; original RxR annotations
  are CC BY 4.0.
- SigLIP2 is Apache-2.0.
- MP3D scenes remain governed by Matterport3D Terms.
- VGGT-1B is CC-BY-NC-4.0.
- The GA-VLN checkpoint/model card does not currently state a license.

Do not upload any external model or dataset into this Git repository.
