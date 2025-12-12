# Superalloy Material Analysis with Deep Learning

## Related Paper

This repository accompanies the paper:

**Deep Learning-Enabled Microstructure-to-Service History Prediction Framework for Nickel-Based Superalloy**

If you use this code in academic work, please cite our paper.

This project implements deep learning models for analyzing superalloy materials, specifically focusing on the heat exposure effects. The system performs both classification and regression tasks using various CNN architectures to predict material properties from microscopic images.

## Project Overview

This repository contains tools and models for analyzing superalloy material images, with a focus on:
- **Classification**: Categorizing material conditions based on temperature and time exposure
- **Regression**: Predicting continuous values such as temperature and exposure duration

## Key Features

- **Multiple CNN Architectures**: 
  - Modified AlexNet
  - Original AlexNet
  - ResNet50
  - Inception V3
- **Preprocessing Pipeline**: Comprehensive image preprocessing tools
- **Data Augmentation**: Advanced augmentation techniques for improved model robustness
- **Performance Visualization**: Confusion matrices, temperature/time comparison plots

## Project Structure

```
├── modify_alex/          # Modified AlexNet implementation
├── original_alex/        # Original AlexNet implementation  
├── inception/            # Inception V3 implementation
├── resnet/              # ResNet50 implementation
├── preprocess/          # Image preprocessing utilities
├── classify/            # Classification task files
├── regression/          # Regression task files
└── spilit.py           # Dataset splitting script
```

## Model Architectures

### Modified AlexNet
- Customized architecture with adjusted kernel sizes and strides
- Feature extraction: 6 convolutional layers with max pooling
- Classifier: 3 fully connected layers with dropout (p=0.5)
- Input size: 256×256×3
- Supports both classification (56 classes) and regression tasks

### ResNet50
- Transfer learning from ImageNet pre-trained weights
- Custom fully connected layers: 2048 → 1024 → 256 → outputs
- Dropout regularization (p=0.5)
- Supports multiple task types: regression, binary classification, multiclass classification

### Inception V3
- Pre-trained on ImageNet with auxiliary logits
- Custom classifier head: 2048 → 1024 → 256 → outputs
- Adaptive activation functions based on task type
- Advanced feature extraction capabilities

## Data Preprocessing Tools

Located in `preprocess/` directory:

- **crop_annotation.py**: Removes annotation regions from images (70px height)
- **delete_green.py**: Filters images with green text markers
- **delete_same.py**: Removes duplicate images
- **divide_square.py**: Splits images into square patches
- **move_tif_jpg.py**: Converts TIFF images to JPEG format
- **count_all_amount.py**: Counts total images in dataset
- **count_everyfile_amount.py**: Counts images per category
- **measure_size.py**: Analyzes image dimensions

## Dataset Organization

The project uses a hierarchical structure for superalloy images:

```
superalloy_data/
├── creep-photos/        # Creep behavior images
├── heat_exposure_photos/# Heat exposure images
├── train/              # Training set (60%)
├── val/                # Validation set (20%)
└── test/               # Test set (20%)
```

Dataset is automatically split using `spilit.py` with a fixed random seed (42) for reproducibility.

## Training

### Classification Task
```python
# Example: Training modified AlexNet for classification
python modify_alex/train.py
```

### Regression Task
```python
# Example: Training for temperature/time regression
python modify_alex/train-r.py
```

## Prediction

### Classification Prediction
```python
python modify_alex/predict.py
```

### Regression Prediction
```python
python modify_alex/predict-r.py
```

The prediction scripts support:
- Batch processing of test images
- Confusion matrix generation
- Error analysis and logging
- Results export to text and Excel formats

## Visualization Tools

Located in `modify_alex/` directory:

- **plot.py**: General classification results plotting
- **plot-r.py**: Regression results visualization
- **plottem.py**: Temperature prediction comparison
- **plottime.py**: Time prediction comparison
- **plotzhu.py**, **plotzhu2.py**, **plotzhu3.py**: Bar chart visualizations

## Performance Metrics

The models generate various performance metrics:
- **Classification**: Confusion matrices, accuracy per class
- **Regression**: MSE, MAE, temperature/time comparison plots
- **Error Analysis**: Detailed error logging with image paths

## Requirements

- Python 3.x
- PyTorch
- torchvision
- PIL (Pillow)
- numpy
- pandas
- matplotlib
- scikit-learn
- tqdm

## GPU Support

All training and prediction scripts support CUDA acceleration:
```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

## Output Files

Training and prediction generate:
- `.pth`: Model checkpoint files (ignored by git)
- `class_indices.json`: Class label mappings
- `confusion_matrix.png/txt/csv`: Performance evaluation
- `predict-result.txt`: Detailed prediction results
- `error_images.txt`: List of misclassified images
- `*.xlsx`: Excel reports with results
- Temperature/time comparison plots

## Model Parameters

Different model configurations are saved in:
- `modify_alex/parameter/`: Modified AlexNet parameters
- `inception/parameter/`: Inception V3 parameters
- `resnet/parameter/`: ResNet50 parameters

Additional specialized parameters:
- `parameter-0C/`: Models for 0°C baseline conditions
- `reg/`: Regression-specific parameters

## Notes

- The `.gitignore` excludes `.pth` files, dataset folders, and certain result directories
- Random seed is set to 42 for reproducible dataset splits
- Image normalization uses mean=(0.5, 0.5, 0.5) and std=(0.5, 0.5, 0.5)
- Default batch size: 32 for training, 4 for validation
- Models support multi-worker data loading for improved efficiency

## License

This project is part of research work on superalloy material analysis.

## Contact

For questions or collaboration, email huojp@mail.tsinghua.edu.cn or sunxb24@mails.tsinghua.edu.cn.
