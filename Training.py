import pyspark
from pyspark.mllib.tree import RandomForest, GradientBoostedTrees
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel, LogisticRegressionWithLBFGS
from pyspark.mllib.util import MLUtils
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession


from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql.functions import col
from pyspark.mllib.linalg import Vectors
from pyspark import SparkContext, SparkConf
from pyspark.sql.session import SparkSession
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml import Pipeline
import numpy as np
# import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


#Starting the spark session
conf = pyspark.SparkConf().setAppName('winequality').setMaster('local')
sc = pyspark.SparkContext(conf=conf)
spark = SparkSession(sc)

#Loading the dataset
df = spark.read.format("csv").load("s3://my-bucket-source-cs643/data/TrainingDataset.csv" , header = True ,sep =";")
df.printSchema()
df.show()

#changing the 'quality' column name to 'label'
for col_name in df.columns[0:-1]+['""""quality"""""']:
    df = df.withColumn(col_name, col(col_name).cast('float'))
df = df.withColumnRenamed('""""quality"""""', "label")


#getting the features and label seperately and converting it to numpy array
features =np.array(df.select(df.columns[0:-1]).collect())
label = np.array(df.select('label').collect())

#creating the feature vector
VectorAssembler = VectorAssembler(inputCols = df.columns[0:-1] , outputCol = 'features')
df_tr = VectorAssembler.transform(df)
df_tr = df_tr.select(['features','label'])

#The following function creates the labeledpoint and parallelize it to convert it into RDD
def to_labeled_point(sc, features, labels, categorical=False):
    labeled_points = []
    for x, y in zip(features, labels):
        lp = LabeledPoint(y, x)
        labeled_points.append(lp)
    return sc.parallelize(labeled_points)

#rdd converted dataset
dataset = to_labeled_point(sc, features, label)

#Splitting the dataset into train and test
training, test = dataset.randomSplit([0.8, 0.2],seed = 5)

#Creating a random forest training classifier
RFmodel = RandomForest.trainClassifier(training, numClasses=10,categoricalFeaturesInfo={}, numTrees=240, maxDepth=30)
# gbt model
GBTmodel = GradientBoostedTrees.trainClassifier(training, categoricalFeaturesInfo={}, numIterations=5)
# Train a naive Bayes model.
naive_model = NaiveBayes.train(training, 1.0)
# lr model
lr_model = LogisticRegressionWithLBFGS.train(training, numClasses=10)



#predictions
predictions = RFmodel.predict(test.map(lambda x: x.features))
#predictionAndLabels = test.map(lambda x: (float(model.predict(x.features)), x.label))
gbt_predictions = GBTmodel.predict(test.map(lambda x: x.features))

#getting a RDD of label and predictions
labelsAndPredictions = test.map(lambda lp: lp.label).zip(predictions)
gbt_labelsAndPredictions = test.map(lambda lp: lp.label).zip(gbt_predictions)

# mevaluation metrics within pyspark
lr_predictionAndLabels = test.map(lambda lp: (float(lr_model.predict(lp.features)), lp.label))
# instantiate metrics object
metrics = MulticlassMetrics(lr_predictionAndLabels)

#overall statistics
f1Score = metrics.fMeasure()
print("LR F1 Score = %s" % f1Score)

# converting to pandas df and finding f1 score
labelsAndPredictions_df = labelsAndPredictions.toDF()
#cpnverting rdd ==> spark dataframe ==> pandas dataframe
labelpred = labelsAndPredictions.toDF(["label", "Prediction"])
labelpred.show()
labelpred_df = labelpred.toPandas()


# naive predictions and accuracy
predictionAndLabel = test.map(lambda p: (naive_model.predict(p.features), p.label))
accuracy = 1.0 * predictionAndLabel.filter(lambda pl: pl[0] == pl[1]).count() / test.count()
print('model accuracy {}'.format(accuracy))
print("LR F1 Score = %s" % f1Score)


#Calculating the F1score
F1score = f1_score(labelpred_df['label'], labelpred_df['Prediction'], average='micro')
print("F1- score: ", F1score)
print(confusion_matrix(labelpred_df['label'],labelpred_df['Prediction']))
print(classification_report(labelpred_df['label'],labelpred_df['Prediction']))
print("Accuracy" , accuracy_score(labelpred_df['label'], labelpred_df['Prediction']))

testErr = gbt_labelsAndPredictions.filter(
    lambda lp: lp[0] != lp[1]).count() / float(test.count())
print('Test Error = ' + str(testErr))


#save training model
RFmodel.save(sc, 's3://my-bucket-source-cs643/data/trainingmodel.model')
