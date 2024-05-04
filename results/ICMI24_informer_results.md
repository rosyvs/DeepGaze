

Pretraining validation metrics: summarized
| metric        |   mean |   min |   max |
|---------------|--------|-------|-------|
| FI: AUROC     |  0.939 | 0.932 | 0.945 |
| CL: Accuracy  |  0.875 | 0.858 | 0.886 |
| PC: MSE (deg) |  0.569 | 0.389 | 0.760 |
| RC: MSE (deg) |  0.745 | 0.464 | 0.977 |

Held-out AUROCs: informer_4task_encoder_varlen_meanpool
| label       |   mean AUROC |   min AUROC |   max AUROC |
|-------------|--------------|-------------|-------------|
| SVT         |        0.523 |       0.464 |       0.563 |
| Rote_X      |        0.661 |       0.629 |       0.673 |
| Rote_Y      |        0.551 |       0.514 |       0.579 |
| Rote_Z      |        0.637 |       0.598 |       0.659 |
| Inference_X |        0.606 |       0.541 |       0.699 |
| Inference_Y |        0.535 |       0.521 |       0.554 |
| Inference_Z |        0.595 |       0.582 |       0.625 |
| Deep_X      |        0.614 |       0.584 |       0.650 |
| Deep_Z      |        0.640 |       0.599 |       0.658 |
| MW          |        0.623 |       0.593 |       0.668 |

Held-out AUROCs: informer_4task_encoder_varlen_maskmeanpool
| label       |   mean AUROC |   min AUROC |   max AUROC |
|-------------|--------------|-------------|-------------|
| SVT         |        0.556 |       0.527 |       0.590 |
| Rote_X      |        0.651 |       0.634 |       0.672 |
| Rote_Y      |        0.536 |       0.500 |       0.566 |
| Rote_Z      |        0.642 |       0.612 |       0.688 |
| Inference_X |        0.621 |       0.558 |       0.705 |
| Inference_Y |        0.525 |       0.505 |       0.546 |
| Inference_Z |        0.607 |       0.578 |       0.630 |
| Deep_X      |        0.618 |       0.583 |       0.648 |
| Deep_Z      |        0.636 |       0.587 |       0.663 |
| MW          |        0.625 |       0.589 |       0.682 |