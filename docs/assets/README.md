# Repository artwork

`social-preview.svg` is the editable source for `social-preview.png`, used in
the main README and as the GitHub social preview. It visualizes the historical
1024 × 1024 timing data in [the benchmark records](../benchmarks/README.md),
not generated model output. Keep the values, speedup and caveats in sync with
that evidence.

Render with librsvg:

```bash
rsvg-convert docs/assets/social-preview.svg -o docs/assets/social-preview.png
```

The PNG is 1280 × 640 with an opaque background. Set it under repository
Settings → General → Social preview → Edit → Upload an image.
