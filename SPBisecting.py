from pyspark.ml.clustering import BisectingKMeans
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

# Loads data.
dataset = spark.read.format("libsvm").load("data/mllib/sample_kmeans_data.txt")

# Trains a bisecting k-means model.
bkm = BisectingKMeans().setK(2).setSeed(1)
model = bkm.fit(dataset)

# Evaluate clustering.
cost = model.computeCost(dataset)
print("Within Set Sum of Squared Errors = " + str(cost))

# Shows the result.
print("Cluster Centers: ")
centers = model.clusterCenters()
for center in centers:
    print(center)
