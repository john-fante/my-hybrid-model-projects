# Flower Detection w/Meta Learning(ViT, CatBoost, SHAP)

First of all, I am very keen on trying new methods. This is why I tried a meta-learning method in this project. In this technique, first I used a pretrained ViT (Vision Transformer) model for the feature extraction stage, then applied PCA for the curse of dimensionality problem, and finally used a tuned CatBoost model for the classification stage.


### ViT Model (for Feature Extraction) -> PCA (for Dimensionality Reduction) -> CatBoostClassifier (for Classification)

|                                 | Training Feature Shape |
|---------------------------------|-------------|
| ViT Features                    | (3038, 64)  |
| After PCA (99 % Variance Ratio) | (3038, 45)  |


I have used the following methods.

* I tried to implementation of distributed deep learning strategy,
* I split the full data into train (3038 images), validation (265 images) and test (367 images),
* I used a pretrained ViT model [1],
* Used <b>tf.data</b> for input pipeline,
* I used a tuned CatBoostClassifier model for classification (tuned with optuna),
* Shap for feature extraction,

## Test Results
![__results___29_1](https://github.com/john-fante/flower-detection-meta-learning/assets/50263592/dd778198-75af-4c51-9ace-de533fcffb91)

## Predictions
![__results___32_0](https://github.com/john-fante/flower-detection-meta-learning/assets/50263592/df0dfe73-dfd3-4506-87fd-55372bb14751)


## References
1. https://github.com/faustomorales/vit-keras
