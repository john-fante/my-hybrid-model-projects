## Smoking Detection w/Hybrid Model (ViT, XGBoost, SHAP)


(kaggle link -> https://www.kaggle.com/code/banddaniel/smoking-detection-w-hybrid-model-vit-xgboost)


I tried a hybrid model in this project. In this technique, first I used a custom ViT (Vision Transformer) model for the feature extraction stage, then applied PCA for the curse of dimensionality problem, and finally used a XGBoost model for the classification stage.



### <span style="color:#e74c3c;">  a pretrained ViT Model (for Feature Extraction) -> PCA (for Dimensionality Reduction) -> a tuned XGBClassifier (for Classification) </span> 


|                                 | Training Feature Shape |
|---------------------------------|-------------|
| ViT Features                    | (716, 64)  |
| After PCA (99 % Variance Ratio) | (716, 50)   |



* I used a mirrored strategy (using 2 T4 GPU at the same time),
* I used a customized ViT model [1],
* Used <b>tf.data</b> for input pipeline,
* I used a XGBoost model for classification (tuned with optuna),
* SHAP for feature explanation,


## My Another Projects
* [Mammals Classification w/Ensemble Deep Learning](https://www.kaggle.com/code/banddaniel/mammals-classification-w-ensemble-deep-learning)
* [Bladder Tissue Classification w/ViT (F1 Scr: 0.82)](https://www.kaggle.com/code/banddaniel/bladder-tissue-classification-w-vit-f1-scr-0-82)
* [Segment Medical Instrument, w/Custom DeepLabv3+(Dice: 0.86)](https://www.kaggle.com/code/banddaniel/segment-medical-instrument-deeplabv3-dice-0-86)
* [Jellyfish Detect (10CV Custom ConvMixer) (F1:0.87)](https://www.kaggle.com/code/banddaniel/jellyfish-detect-10cv-custom-convmixer-f1-0-87)


## References
1. https://github.com/faustomorales/vit-keras
