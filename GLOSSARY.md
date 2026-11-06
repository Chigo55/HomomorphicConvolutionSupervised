# Homomorphic Convolution Supervised

Supervised low-light image enhancement research: training a model on paired low/normal-light images and reporting its quality on public benchmarks.

## Images

**Low-light image**:
An under-exposed input image.
_Avoid_: low, dark image

**Normal-light image**:
The properly exposed image paired with a Low-light image; the ground truth.
_Avoid_: high, target, GT

**Enhanced image**:
The model's output for a Low-light image.
_Avoid_: pred, output

**Illumination**:
The low-frequency component of an image's luminance in the homomorphic decomposition.
_Avoid_: illuminance

**Reflectance**:
The high-frequency (detail) component of an image's luminance in the homomorphic decomposition.

## Data splits

**Training set**:
Paired images the model is fitted on.

**Validation set**:
Paired images used to pick the best Checkpoint and to stop training early.

**Benchmark set**:
Public paired datasets on which Benchmark results are reported, one dataset at a time.
_Avoid_: test set

**Inference set**:
Images that are only enhanced and saved, never scored.

## Experiment tracking

**Experiment**:
A group of Runs that answer one research question or hypothesis (e.g. a cutoff ablation).
_Avoid_: project, version, "train"

**Run**:
One training execution with exactly one configuration and one seed.
_Avoid_: experiment, version, trial

**Baseline**:
The Run with the default configuration and no Experiment override; every Experiment is compared against it.
_Avoid_: reference model, default run

**Checkpoint**:
A snapshot of a Run's weights at one point in training.
_Avoid_: model file, weights

**Benchmark result**:
The metrics of one Checkpoint evaluated on one dataset of the Benchmark set.
_Avoid_: test result, score
