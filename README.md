# Natural Scene Image Classification (Computer Vision)

## What this project does
This project trains an image classifier that predicts the scene class for a given photo.  
Classes: buildings, forest, glacier, mountain, sea, street.

It is designed as a practical Computer Vision portfolio project: training, evaluation on a held out labeled test set, and a small inference script for single image predictions.

## Dataset
Intel Image Classification (Natural Scenes) dataset.

Folder structure used by this repo:

```
data/
  seg_train/seg_train/
    buildings/
    forest/
    glacier/
    mountain/
    sea/
    street/
 
  seg_pred/seg_pred/
    (unlabeled images for demo predictions)
```

Splits and sizes from the training run:

1. Train split (from seg_train): 11,929 images  
2. Test split (seg_test): 3,000 images  

## Approach
Model: ResNet18 with ImageNet pretrained weights, fine tuned for 6 classes.

Training strategy (two phase fine tuning):

1. Warm up: freeze the backbone, train only the classifier head for 2 epochs  
2. Fine tune: unfreeze the last ResNet block plus classifier head for 3 epochs  

Why two phase fine tuning:
1. Faster and more stable optimization on CPU
2. Improves accuracy compared with training only the head
3. Reduces overfitting risk compared with unfreezing everything immediately

Data transforms:

1. Train time augmentations: horizontal flip, small rotation, light color jitter
2. Eval time transforms: resize plus normalization only

## Results
Best validation accuracy: 0.9292 (saved checkpoint)

Test set performance (3,000 images):

1. Accuracy: 0.927
2. Macro average precision: 0.927
3. Macro average recall: 0.929
4. Macro average F1: 0.928

Per class highlights:

1. Strongest class: forest (F1 0.982)
2. Most challenging class: glacier (F1 0.887), likely due to visual overlap with mountain and sea

The script saves these evaluation artifacts:

1. Confusion matrix image: `outputs/confusion_matrix_test.png`
2. Text classification report: `outputs/test_classification_report.txt`
3. Training history: `outputs/train_history.json`

## How to run

### Install dependencies
CPU install:

```bash
pip install torch torchvision torchaudio
pip install matplotlib scikit-learn pillow
```

### Train and evaluate
From the project root:

```bash
python src/train.py
```

This will:
1. Train the model with a train validation split created from seg_train
2. Save the best validation checkpoint to `models/resnet18_intel_scenes.pt`
3. Evaluate on the labeled test set in seg_test
4. Save the confusion matrix and report to `outputs/`

### Predict on one image
After training, run:

```bash
python src/predict.py "path input"
```

The `predict.py` script prints a single final predicted class with a confidence score (top 1 prediction).

## Repository structure
```
cv_natural_scenes/
  data/            (local only, not recommended to commit)
  src/
    train.py
    predict.py
  models/          (model checkpoint saved here)
  outputs/         (plots and reports saved here)
  README.md
```

## Production style considerations
If deployed in a real product, I would monitor:

1. Input drift: image resolution, brightness, compression artifacts, camera source changes
2. Confidence drift: shifts in predicted probability distribution
3. Class distribution drift: spikes in certain classes that might indicate data shift
4. Accuracy sampling: periodically label a small slice of recent images to measure real world performance
5. Retraining triggers: drift thresholds or periodic retraining cadence

## Reproducibility notes
This dataset is cleaner than many real world image pipelines. Real world performance can drop due to new camera conditions, new scene types, and domain shift.  
Saving the best validation checkpoint helps avoid overfitting when fine tuning.
