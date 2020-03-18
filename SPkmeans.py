from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler

import sys

args =sys.argv
filename=args[1]
print(filename)
sc = SparkContext("local", "first app")


spark = SparkSession \
    .builder \
    .appName("Analyzing London crime data") \
    .getOrCreate()
# Loads data.
dataset = spark.read.load(filename, format="csv", sep=",", inferSchema="True", header="false")
k = 8
n=len(dataset.columns)
ToBeDropped = n-1
col=dataset.columns
dataset=dataset.drop(col[ToBeDropped ])  
dataset.show()
# Trains a k-means model.
cols = dataset.columns

assb = VectorAssembler(
    inputCols=cols,
    outputCol="features")
dataset = assb.transform(dataset)
dataset.show()
kmeans = KMeans().setK(k).setSeed(1).setDistanceMeasure("cosine")
model = kmeans.fit(dataset)

# Make predictions
predictions = model.transform(dataset)

# Evaluate clustering by computing Silhouette score
evaluator = ClusteringEvaluator()

silhouette = evaluator.evaluate(predictions)
print("Silhouette with squared euclidean distance = " + str(silhouette))

# Shows the result.
centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)
