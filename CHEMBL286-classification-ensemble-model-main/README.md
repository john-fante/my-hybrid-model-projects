
# ChEMBL286 Classification with an Ensemble Model(CNN + CatBoostClassifier)

I have tried 3 models for ChEMBL286 dataset[1,2] (renin enzyme[3]).
Firstly, I used a basic CNN model, then I used a CNN model for feature extraction and a CatBoost model for classification. Lastly, I combined two models. 

I have used the following methods.

* A basic CNN model (Model 1)
* A CNN model with elu activation function for the feature extraction, and CatBoostClassifier for the classification stage (Model 2)
* Custom convolution block
* Custom callback class  that used predicting a sample from the validation in dataset during training
* Weighted ensemble method for the last prediction (Model 3)

## Results
* Classification metrics

<br>

|                            | Accuracy | Precision* | Recall* | F1-Score* |
|----------------------------|----------|-----------|--------|----------|
| Model1 (CNN)               | 82 %     | 0.82      | 0.8    | 0.8      |
| Model2 (CNN + CatBoost)    | 78 %     | 0.78      | 0.75   | 0.76     |
| Model3 (Weighted Ensemble) | 85 %     | 0.86      | 0.83   | 0.84     |

<i>* macro average</i>

* Predictions during training


https://github.com/john-fante/CHEMBL286-classification-ensemble-model/assets/50263592/fbec9566-28aa-443d-967e-82dadf8dc2a2


<br>

* Base CNN model results

![dmafngt](https://github.com/john-fante/CHEMBL286-classification-ensemble-model/assets/50263592/17ff6f83-782d-45d9-a331-43b9cf6a92d3)

<br>

* Confusion matrices
![New Project (2)](https://github.com/john-fante/CHEMBL286-classification-ensemble-model/assets/50263592/0708ff58-1556-4cb5-a28b-9816de561cf9)


## References
1. https://github.com/cansyl/DEEPscreen
2. https://www.ebi.ac.uk/chembl/target_report_card/CHEMBL286/
3. https://my.clevelandclinic.org/health/body/22506-renin#:~:text=Renin%20is%20an%20enzyme%20that,blood%20pressure%20drops%20too%20low.
