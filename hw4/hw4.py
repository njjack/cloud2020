
# Add Spark Python Files to Python Path
import sys
import os
SPARK_HOME = "/spark" # Set this to wherever you have compiled Spark
os.environ["SPARK_HOME"] = SPARK_HOME # Add Spark path
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1" # Set Local IP
sys.path.append( SPARK_HOME + "/python") # Add python files to Python Path

import numpy as np
from pyspark import SparkConf, SparkContext
import pyspark

def getSparkContext():
    """
    Gets the Spark Context
    """
    conf = (SparkConf()
         .setMaster("local") # run on local
         .setAppName("Logistic Regression") # Name of App
         .set("spark.executor.memory", "1g")) # Set 1 gig of memory
    sc = SparkContext(conf = conf) 
    return sc

def mapper(line):
    """
    Mapper that converts an input line to a feature vector
    """    
    feats = line.strip().split(",") 
    # labels must be at the beginning for LRSGD
    label = feats[len(feats) - 1] 
    feats = feats[: len(feats) - 1]
    feats.insert(0,label)
    features = [ float(feature) for feature in feats ] # need floats
    return np.array(features)

def predict(w, b, point):
    y = int(point.item(0))
    x = point.take(range(1, point.size))
    yhat = np.dot(w, x) + b
    return 1.0 / (1.0 + np.exp(-yhat)) 

def gradient(w, b, point):
    y = int(point.item(0))
    x = point.take(range(1, point.size))
    t = predict(w, b, point)-y
    w = t * x
    b = t
    return w, b

sc = getSparkContext()

# Load and parse the data
data = sc.textFile("data_banknote_authentication.txt")
parsedData = data.map(mapper)

lr = 0.1 #learning rate
epoch = 500
w = np.zeros(parsedData.first().size - 1)
b = 0
for i in range(epoch):
            data_random = parsedData.sample(True, 0.2, i)
            count = data_random.count()
            g_w = data_random.map(lambda x: gradient(w, b, x)[0]).reduce(lambda x, y: x + y)
            g_b = data_random.map(lambda x: gradient(w, b, x)[1]).reduce(lambda x, y: x + y)

            w = w - lr * g_w / count
            b = b - lr * g_b / count

labelsAndPreds = parsedData.map(lambda point: (int(point.item(0)), round(predict(w, b, point))))
error = labelsAndPreds.filter(lambda seq: seq[0] != seq[1]).count() / float(parsedData.count())
print("error = " + str(error))