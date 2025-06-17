#!/usr/bin/env python
# coding: utf-8

# In[ ]:


dividedData = data.randomSplit([0.7, 0.3]) 
trainingData = dividedData[0] #index 0 = data training
testingData = dividedData[1] #index 1 = data testing
train_rows = trainingData.count()
test_rows = testingData.count()
print ("Training data rows:", train_rows, "; Testing data rows:", test_rows)


# In[ ]:


tokenizer = Tokenizer(inputCol="text", outputCol="SentimentWords")
tokenizedTrain = tokenizer.transform(trainingData)
tokenizedTrain.show(truncate=False, n=5)


# In[ ]:


swr = StopWordsRemover(inputCol=tokenizer.getOutputCol(), 
                       outputCol="MeaningfulWords")
SwRemovedTrain = swr.transform(tokenizedTrain)
SwRemovedTrain.show(truncate=False, n=5)


# In[ ]:


hashTF = HashingTF(inputCol=swr.getOutputCol(), outputCol="features")
numericTrainData = hashTF.transform(SwRemovedTrain).select(
    'label', 'MeaningfulWords', 'features')
numericTrainData.show(truncate=False, n=3)


# In[ ]:


lr = LogisticRegression(labelCol="label", featuresCol="features", 
                        maxIter=200, regParam=0.01)
model = lr.fit(numericTrainData)
predictions_test = model.transform(numericTest)
print ("Training is done!")


# In[ ]:


lrtr = LogisticRegression(labelCol="label", featuresCol="features", 
                        maxIter=200, regParam=0.01)
modeltr = lr.fit(numericTrainData)
predictions_testtr = model.transform(numericTrainData)
print ("Training is done!")


# In[ ]:


prediction = modeltr.transform(numericTrainData)
predictionFinaltr = prediction.select(
    "MeaningfulWords", "prediction", "Label", "probability", "rawPrediction")
predictionFinaltr.show(n=4, truncate = False)
correctPrediction = predictionFinaltr.filter(
    predictionFinaltr['prediction'] == predictionFinaltr['Label']).count()
totalData = predictionFinaltr.count()
print("correct prediction:", correctPrediction, ", total data:", totalData, 
      ", accuracy:", correctPrediction/totalData)


# In[ ]:


import matplotlib.pyplot as plt

# Create a Pipeline estimator and fit on train DF, predict on test DF
predictions = model.transform(numericTest)

# Returns as a list (false positive rate, true positive rate)
preds = predictions.select('label','probability').rdd.map(lambda row: (float(row['probability'][1]), float(row['label'])))
points = CurveMetrics(preds).get_curve('roc')

plt.figure()
x_val = [x[0] for x in points]
y_val = [x[1] for x in points]

plt.plot(x_val, y_val)


# In[ ]:


prediction = model.transform(numericTrainData)
predictionFinal = prediction.select(
    "MeaningfulWords", "prediction", "Label", "probability", "rawPrediction")
predictionFinal.show(n=4, truncate = False)
correctPrediction = predictionFinal.filter(
    predictionFinal['prediction'] == predictionFinal['Label']).count()
totalData = predictionFinal.count()
print("correct prediction:", correctPrediction, ", total data:", totalData, 
      ", accuracy:", correctPrediction/totalData)


# In[ ]:


from pyspark.ml.classification import RandomForestClassifier

model2 = RandomForestClassifier(
    numTrees=3, maxDepth=5, seed=42, labelCol="label",featuresCol="features")
model2 = model2.fit(numericTrainData)
predictions_test_rfc = model2.transform(numericTest)
print ("Model is trained!")


# In[ ]:


predictionFinal2 = model2.transform(numericTrainData)
predictionFinal_rfc = prediction.select(
    "MeaningfulWords", "prediction", "Label", "probability")
predictionFinal_rfc.show(n=4, truncate = False)
correctPrediction_rfc = predictionFinal_rfc.filter(
    predictionFinal_rfc['prediction'] == predictionFinal_rfc['Label']).count()
totalData = predictionFinal_rfc.count()
print("correct prediction:", correctPrediction_rfc, ", total data:", totalData, 
      ", accuracy:", correctPrediction_rfc/totalData)


# In[ ]:


# Create a Pipeline estimator and fit on train DF, predict on test DF
predictions = model2.transform(numericTrainData)

# Returns as a list (false positive rate, true positive rate)
preds = predictions.select('label','probability').rdd.map(lambda row: (float(row['probability'][1]), float(row['label'])))
points = CurveMetrics(preds).get_curve('roc')

plt.figure()
x_val = [x[0] for x in points]
y_val = [x[1] for x in points]

plt.plot(x_val, y_val)


# In[ ]:


from pyspark.ml.classification import DecisionTreeClassifier
dtc = DecisionTreeClassifier(maxDepth=3, labelCol="label", featuresCol="features")
model3 = dtc.fit(numericTrainData)
print ("Training is done!")


# In[ ]:


prediction = model3.transform(numericTrainData)
predictionFinaldtc = prediction.select(
    "MeaningfulWords", "prediction", "Label", "probability")
predictionFinaldtc.show(n=4, truncate = False)
correctPrediction = predictionFinaldtc.filter(
    predictionFinaldtc['prediction'] == predictionFinaldtc['Label']).count()
totalData = predictionFinaldtc.count()
print("correct prediction:", correctPrediction, ", total data:", totalData, 
      ", accuracy:", correctPrediction/totalData)


# In[ ]:


# Create a Pipeline estimator and fit on train DF, predict on test DF
predictions = model3.transform(numericTest)

# Returns as a list (false positive rate, true positive rate)
preds = predictions.select('label','probability').rdd.map(lambda row: (float(row['probability'][1]), float(row['label'])))
points = CurveMetrics(preds).get_curve('roc')

plt.figure()
x_val = [x[0] for x in points]
y_val = [x[1] for x in points]

plt.plot(x_val, y_val)


# In[21]:


from pyspark.ml.classification import FMClassifier
# Train a FM model.
fm = FMClassifier(labelCol="label", \
                  featuresCol="features", \
                  stepSize=0.001, maxIter=200)
model4 = fm.fit(numericTrainData)
print ("Training is done!")

