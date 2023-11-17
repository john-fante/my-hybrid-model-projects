# Skin Cancer Detection w/Meta Learning (CNN, CatBoost)

First of all, I am very keen on trying new methods. This is why I tried a meta-learning method in this project. In this technique, first I used a custom ConvMixer block based CNN model for the feature extraction stage, then applied PCA for the curse of dimensionality problem, and finally used a CatBoost model for the classification stage. 

In this dataset, there are several images of the same patient. It is a cautious issue when splitting data. If we directly shuffle and split this data, this will give rise to the overfitting problem, because our model saw the testing data. This is why I used the last 298 images for testing.

I dropped a feature named 'biopsed' because this feature is very correlated with the target variable. If I use this feature, the test accuracy is nearly 70 %.

<i><b>Version 2: </b> There was a problem with shuffling training data and I fixed it. I tried the 0.98 % variance ratio in the PCA stage. And I added the test set prediction examples. </i>


![download (40)](https://github.com/john-fante/skin-cancer-detect-meta-learning/assets/50263592/5532e114-02ba-4a39-a774-658e8fd80018)


I have used the following methods.

* I used a model created with <b>ConvMixer </b> blocks [1,2],
* <b>gelu</b> activation function during the feature extraction stage,
* Used <b>tf.data</b> for input pipeline,
* I split the full data into train (2000 images) and test (298 images),

## Test Results
![__results___40_1](https://github.com/john-fante/skin-cancer-detect-meta-learning/assets/50263592/51294fdf-6f13-4748-a648-e5360f9c717f)

## Test Predictions
![__results___43_2](https://github.com/john-fante/skin-cancer-detect-meta-learning/assets/50263592/b0995c73-9d08-41e9-8c78-ce5dfbb542d2)
![__results___43_0](https://github.com/john-fante/skin-cancer-detect-meta-learning/assets/50263592/389256ad-20db-4b30-9550-07223865504f)


## References
1. Trockman, A., & Kolter, J. Z. (2022). Patches Are All You Need? (Version 1). arXiv. https://doi.org/10.48550/ARXIV.2201.09792
2. https://keras.io/examples/vision/convmixer
