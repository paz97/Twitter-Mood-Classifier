# Twitter-Mood-Classifier Documentation
Multiclass Mood Classifier using Twitter Dataset


# Installation

Clone the repo using ```git clone https://github.com/paz97/Twitter-Mood-Classifier.git```

This project is made using ```Python 3```. The following dependencies need to be installed on your local machine:
```
pandas
numpy
sklearn
seaborn
matplotlib
```

Run the program using ```python3 analysis.py```

# Overview

This project uses the SemEval 2018 Twitter dataset to perform multiclass classification. Training was done on 4 classifiers and the results were as follows:

| Classifier | Accuracy |
| ---        |  --- | 
LinearSVC     |  0.874273
LogisticRegression    |    0.884779
MultinomialNB         |    0.852905
RandomForestClassifier  |  0.407241

![alt text](https://github.com/paz97/Twitter-Mood-Classifier/blob/master/classifier_comparison.png)

Since Logistic Regression gave the best results, that was used for testing.

The final metrics using this classifier were as follows:

|  |  precision  |  recall | f1-score  | support |
| --- | --- | --- | --- | --- |
|  anger    |   0.42   |   0.70    |  0.53   |   1002 |
|  fear    |   0.59   |   0.33  |    0.42   |    986 |
|   joy    |   0.67    |  0.65   |   0.66   |   1105 |
|  sadness     |  0.49     | 0.40   |   0.44   |    975 |
|  micro avg      | 0.52  |    0.52   |   0.52   |   4068 |
|  macro avg    |   0.54   |   0.52   |   0.51   |   4068 |
| weighted avg   |    0.55    |  0.52   |   0.52  |    4068 |

# References
- https://competitions.codalab.org/competitions/17751 
- https://github.com/cbaziotis/ntua-slp-semeval2018
- https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f
 
