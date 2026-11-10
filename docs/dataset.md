# COCO 2014 dataset preparation

The scripts read captions from the COCO 2014 validation annotation file. The
expected JSON structure contains an `annotations` array whose elements have a
`caption` field.

## Required caption file

Download and extract the COCO 2014 annotations so that the following file is
available:

```text
annotations/captions_val2014.json
```

Set `coco_annotation_path` near the top of every script you run. The checked-in
value is a machine-specific example and will not work on another computer.

## Metric-specific data

CLIP uses only captions and generated images. LPIPS first creates high-step
reference images from the same prompts and seeds. FID needs a directory of
real COCO validation images in addition to captions; review the `base_path`,
source-image path, and real-image directory in the selected FID script.

## Sampling behavior

The scripts de-duplicate captions, sort them for deterministic ordering, and
select the requested number. Random seeds are fixed to 42. Changing the image
count changes the evaluated caption subset and should be recorded with the
result.

## Storage planning

FID defaults to 10,000 images for each step setting, while CLIP and LPIPS
typically use 1,000. Generated PNGs and reference directories can therefore
consume significant disk space. Keep generated artifacts outside version
control and retain the configuration and CSV summary needed to reproduce them.

## Example directory layout

```text
coco2014/
|-- annotations/
|   `-- captions_val2014.json
`-- val2014/
    |-- COCO_val2014_000000000042.jpg
    `-- ...
```

## Data required by each metric

| Metric | Required data |
| --- | --- |
| CLIP | COCO captions |
| LPIPS | COCO captions and generated high-step reference images |
| FID | COCO captions and real COCO validation images |
| Latency | COCO captions |

Each experiment script must point to the correct local dataset path.
