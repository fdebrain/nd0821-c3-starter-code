# Model card

## Model Details

Frédéric Debraine created the model on November 29th 2021.
The model is a Random Forest Classifier using Scikit-Learn's 1.0.1 default hyperparameters. This is version 1.0.

## Intended Use

This model should be used to predict whether income exceeds $50K/yr based on census data such as age, gender, education, etc...

## Training and Evaluation Data

The data was obtained from the Census Income Dataset on [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/census+income).
The target class consists of the 'salary' column whose value can either be `>50K` or `<=50K`.

The original data has 48842 observations and was broken into a train set and an evaluation set using a 80-20 split without any stratification.
Before training the model, a One Hot Encoder was used on the categorical features and a label binarizer was applied on the labels.

## Metrics

The model was evaluated using precision, recall and F-beta (beta=1). They are estimated to be 0.73, 0.61 and 0.67 respectively.

## Ethical Considerations

The model requires personal data as input.
We did not perform any bias analysis.

## Caveats and Recommendations

There is a significant label imbalance with only 25% of observations having a salary higher than $50K/yr.
Also most of the training data was collected from working people, born in the USA and from Caucasian ethnicity.
The Census Income Dataset was collected in 1994. Therefore, we cannot guarantee the performance of the model in a real-world scenario. Further evaluation of our model on other groups of people is required.
