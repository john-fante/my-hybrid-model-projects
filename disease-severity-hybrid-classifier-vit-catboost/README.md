## Disease Severity Hybrid Classifier (ViT,CatBoost)

(kaggle link -> https://www.kaggle.com/code/banddaniel/disease-severity-hybrid-classifier-vit-catboost )

I tried a hybrid model in this project. In this technique, first I used a custom ViT (Vision Transformer) model for the feature extraction stage, then applied PCA for the curse of dimensionality problem, and finally used a CatBoost model for the classification stage.


![download (34)](https://github.com/john-fante/my-hybrid-model-projects/assets/50263592/285a4b93-5c0d-4ae0-a51d-457c9f723259)


* The project took place using Google TPU,
* I used a customized ViT model [1],
* Used <b>tf.data</b> for input pipeline,
* I used a CatBoost model for classification,


## References
1. https://github.com/faustomorales/vit-keras
