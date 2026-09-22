# Pinned runtime inputs

Only two external assets are retained:

| Asset | Why it is included | Upstream license |
| --- | --- | --- |
| `wheels/diffusers-0.41.0.dev0-py3-none-any.whl` | Exact Qwen-Image-2.1 pipeline and transformer source used by the guarded kernels | Apache-2.0 |
| `sources/sageattention-d1a57a5.tar.gz` | Optional reproducible SageAttention2 build from revision `d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5` | Apache-2.0 |

Run `python scripts/verify_assets.py` to check [manifest.json](manifest.json).
Installers verify the asset they use. The full Diffusers upstream commit was
not recorded; the wheel checksum and individual validated source checksums are
the available provenance. Do not substitute a same-version wheel without
revalidating the kernels. Both assets include their upstream licenses, also
copied to [licenses/](../licenses/).

The wheel is about 5.8 MB and the source archive about 64 KB. They are ordinary
Git binary files, not LFS pointers. Duplicate extracted sources, prebuilt Sage
wheels and experimental snapshots are excluded. Project distributions exclude
these assets; use the repository setup scripts to install the runtime inputs.
