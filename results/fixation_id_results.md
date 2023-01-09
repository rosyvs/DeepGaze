## Fixation Experiments

### Experiment 1: Fixation ID (Binary Classification Fixation=1 or Saccade=0)

**Experiment Details**

**Fixation Percentage in Data**

Fixation Percentage in our Data: 80% Fixations

Fixation Percentage in their data: ~90% Fixations

**Class Weights** (calculated using inverse ratio of the data)

Class Weights Ours: [Saccades: 3.86, Fixations: 0.26]

Class Weights Theirs: [Saccades: 4.53, Fixations: 0.18]]

**Results Tables**

Table 1: Using Class Weights (Saccades: 3.86, Fixations: 0.26)

| Experiment                  | Train Loss | Test Loss | Test AUROC |
|-----------------------------|------------|----------|-----------|
| Convolutions+GRU Pretrained | 0.43       | 0.44     | 0.68      |
| Convolutions+GRU            | 0.44       | 0.44     | 0.65      |
| GRU                         | 0.44       | 0.44     | 0.65      |

Table 2. Using Class Weights (Saccades: 3.0, Fixations: 1.0)

| Experiment                  | Train Loss | Val Loss | Val AUROC |
|-----------------------------|------------|----------|-----------|
| Convolutions+GRU Pretrained | 0.56       | 0.56     | 0.72      |
| Convolutions+GRU            | 0.557      | 0.56     | 0.72      |

Table 3. Sequence Length Differences (Class Weights = Saccades: 3.0, Fixations: 1.0)

| Experiment                      | Sequence Length | Train Loss | Val Loss | Val AUROC |
|---------------------------------|-----------------|------------|----------|-----------|
| Convolutions+GRU Not Pretrained | 500             | 0.56       | 0.56     | 0.72      |
| Convolutions+GRU Not Pretrained | 50              | 0.27       | 0.27     | 0.92      |

Table 4. Model Capacities vs Sequence Lengths (Trained 10 Epochs)

| Model    | GRU Hidden Size | Sequence Length | Val Accuracy | Val AUROC |
|----------|-----------------|-----------------|--------------|-----------|
| Conv+GRU | 128             | 100             | 0.82         | 0.86      |
| Conv+GRU | 256             | 100             | 0.81         | 0.86      |
| Conv+GRU | 128             | 250             | 0.76         | 0.74      |
| Conv+GRU | 256             | 250             | 0.77         | 0.73      |
| Conv+GRU | 128             | 500             | 0.74         | 0.66      |

