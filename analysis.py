import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn import metrics



def processData(path, sentiment):
    cols = ['ID', 'Tweet', 'Affect', 'Intensity']
    df = pd.read_table(path, skiprows = 1, names = cols)
    df['Val'] = np.nan
    for index, row in df.iterrows():
        if '0' in row['Intensity']:
            df.at[index,'Val'] = 0
        elif '1' in row['Intensity']:
            df.at[index,'Val'] = 1
        elif '2' in row['Intensity']:
            df.at[index,'Val'] = 2
        else:
            df.at[index,'Val'] = 3

    df = df[df.Val > 1]
    return df

def combinedList(anger, fear, joy, sad):
    return pd.concat([anger, fear, joy, sad])


def getDf(dataset):
    if dataset == 'train':
        anger = processData('data/training/EI-oc-En-anger-train.txt', 'anger')
        fear = processData('data/training/EI-oc-En-fear-train.txt', 'fear')
        joy = processData('data/training/EI-oc-En-joy-train.txt', 'joy')
        sad = processData('data/training/EI-oc-En-sadness-train.txt', 'sad')

    else:
        anger = processData('data/test/2018-EI-oc-En-anger-test.txt', 'anger')
        fear = processData('data/test/2018-EI-oc-En-fear-test.txt', 'fear')
        joy = processData('data/test/2018-EI-oc-En-joy-test.txt', 'joy')
        sad = processData('data/test/2018-EI-oc-En-sadness-test.txt', 'sad')

    retDf = combinedList(anger, fear, joy, sad)
    retDf['sentId'] = retDf['Affect'].factorize()[0]
    return retDf


def main():
    trainDf = getDf('train')
    sent_id_df = trainDf[['Affect', 'sentId']].drop_duplicates().sort_values('sentId')
    sent_to_id = dict(sent_id_df.values)
    id_to_sent = dict(sent_id_df[['sentId','Affect']].values)



    tfidf = TfidfVectorizer(sublinear_tf = True, min_df = 5, norm = 'max',
            encoding = 'latin-1', ngram_range = (1,2), stop_words = 'english')

    features = tfidf.fit_transform(trainDf.Tweet).toarray()
    labels = trainDf.sentId

    N = 2
    for Affect, sentId in sorted(sent_to_id.items()):
      features_chi2 = chi2(features, labels == sentId)
      indices = np.argsort(features_chi2[0])
      feature_names = np.array(tfidf.get_feature_names())[indices]
      unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
      bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
      # print("# '{}':".format(Affect))
      # print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
      # print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))



    models = [
        RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
        LinearSVC(),
        MultinomialNB(),
        LogisticRegression(random_state=0),
    ]

    CV = 5
    cv_df = pd.DataFrame(index=range(CV * len(models)))
    entries = []
    for model in models:
        model_name = model.__class__.__name__
        accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
        for fold_idx, accuracy in enumerate(accuracies):
            entries.append((model_name, fold_idx, accuracy))
    cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])



    sns_plot = sns.boxplot(x='model_name', y='accuracy', data=cv_df)
    sns_plot = sns.stripplot(x='model_name', y='accuracy', data=cv_df,
              size=8, jitter=True, edgecolor="gray", linewidth=2)
    fig = sns_plot.get_figure()
    fig.savefig("classifier_comparison.png")

    f = open("classifier_accuracy_comparison.txt","w+")
    f.write(str(cv_df.groupby('model_name').accuracy.mean()))
    f.close()
    #print(cv_df.groupby('model_name').accuracy.mean())

    model = LogisticRegression()
    X_train = features
    y_train = labels
    model.fit(X_train, y_train)

    testDf = getDf('test')
    X_test = features = tfidf.transform(testDf.Tweet).toarray()
    y_test = labels = testDf.sentId
    y_pred = model.predict(X_test)
    #print(model.score(features,labels))
    conf_mat = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(10,10))
    sns_plot2 = sns.heatmap(conf_mat, annot=True, fmt='d',
                xticklabels=sent_id_df.Affect.values, yticklabels=sent_id_df.Affect.values)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig("test_confusion_matrix.png")

    f = open("metrics.txt","w+")
    f.write(metrics.classification_report(y_test, y_pred, target_names=testDf['Affect'].unique()))
    f.close()
    #print(metrics.classification_report(labels, y_pred, target_names=testDf['Affect'].unique()))


if __name__ == "__main__":
    main()
