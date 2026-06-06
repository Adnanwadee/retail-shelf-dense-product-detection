# DINOv2 + Mini-Batch K-Means Visual Pseudo-Labeling Overview

This document describes the unsupervised visual pseudo-labeling stage used in the retail shelf dense product detection project.

## Why this stage was needed

The original detection dataset provided bounding boxes for a single generic product class.

That means the detection task could localize products, but it could not distinguish product categories or visual groups directly from the original annotations.

To extend the project beyond one-class detection without manual relabeling, a visual grouping pipeline was introduced.

## Core idea

Product instances were cropped from shelf images using available bounding boxes. Each crop was then transformed into a visual embedding using DINOv2. These embeddings were clustered with Mini-Batch K-Means to generate visual product-group pseudo-labels.

```text
Shelf image
  → Product bounding boxes
  → Product crop extraction
  → Crop filtering
  → DINOv2 visual embedding extraction
  → Mini-Batch K-Means clustering
  → Cluster review using mosaic visualization
  → 91 visual product-group pseudo-labels
```

## Why DINOv2

DINOv2 was used because the problem depends heavily on visual similarity rather than text semantics.

Important visual cues include:

- Package shape
- Texture
- Color layout
- Edges
- Object size
- Fine-grained visual appearance
- Packaging patterns

This made DINOv2 more suitable than text-image semantic matching approaches when true SKU names or captions were not available.

## Pseudo-label meaning

The generated labels are visual product groups, not verified commercial SKU identities.

Correct interpretation:

```text
Visual product-group pseudo-labels
```

Incorrect interpretation:

```text
Verified SKU labels
Commercial product identity labels
Billing-ready product classes
```

## Output

The pseudo-labeling process produced 91 visual product groups.

These groups were later used as optional supervision for a ResNet-50 visual product-group classifier.

## Quality control

Cluster quality was reviewed visually using mosaic visualization to inspect whether visually similar crops were grouped together.

## Public repository note

The full intermediate crop dataset and generated archive are not included in this repository to avoid publishing large derived data files and to keep the repository lightweight.

The public version documents the method and includes the cleaned downstream classification notebook.

## Limitations

- Visual groups may not map one-to-one to real commercial SKUs.
- Similar packages may be grouped together even if they represent different products.
- Different appearances of the same product may fall into separate visual groups.
- Real SKU-level recognition would require verified product labels and a product database.
