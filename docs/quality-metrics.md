# Image-quality metrics

DIFS evaluates three complementary properties of generated images. No single
metric fully represents perceptual quality, so interpret them together with
latency and experiment settings.

## CLIP score

CLIP embeds each prompt and generated image into a shared representation. The
scripts compute cosine similarity and average it across the evaluated prompts.
Higher values indicate stronger text–image alignment under the selected CLIP
model.

Run an available model script, for example:

```bash
python accuracy_est/CLIP/v1.5_CLIP.py
```

## FID

Fréchet Inception Distance compares feature distributions for real and
generated image sets. Lower values indicate that the generated distribution is
closer to the real-image distribution. FID is sensitive to sample count,
preprocessing, and the exact reference set, so comparisons require identical
protocols.

```bash
python accuracy_est/FID/v1.5_FID.py
```

## LPIPS

LPIPS measures perceptual distance between paired images. These experiments
use high-step generations as references and compare lower-step generations
created from matching prompts and deterministic seeds. Lower values indicate
greater similarity to the reference generation.

```bash
python accuracy_est/LPIPS/v1.5_LPIPS.py
```

## Interpreting a step sweep

Increasing denoising steps usually increases latency, but quality metrics may
improve unevenly or plateau. Plot each metric against measured latency rather
than assuming that step count has the same cost across models and resolutions.

Record the model identifier, model revision, resolution, seed policy, prompt
subset, sample count, step list, and metric implementation with every result.
