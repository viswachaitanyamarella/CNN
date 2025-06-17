#!/usr/bin/env python
# coding: utf-8

# In[ ]:


tokenizedTest = tokenizer.transform(testingData)
SwRemovedTest = swr.transform(tokenizedTest)
numericTest = hashTF.transform(SwRemovedTest).select(
    'Label', 'MeaningfulWords', 'features')
numericTest.show(truncate=False, n=2)


# In[ ]:


prediction = model.transform(numericTest)
predictionFinal = prediction.select(
    "MeaningfulWords", "prediction", "Label", "probability", "rawPrediction")
predictionFinal.show(n=4, truncate = False)
correctPrediction = predictionFinal.filter(
    predictionFinal['prediction'] == predictionFinal['Label']).count()
totalData = predictionFinal.count()
print("correct prediction:", correctPrediction, ", total data:", totalData, 
      ", accuracy:", correctPrediction/totalData)


# In[ ]:


import sklearn 

y_true = predictionFinaltr.select(['label']).collect()
y_pred = predictionFinaltr.select(['prediction']).collect()

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_true, y_pred))


# In[ ]:


import sklearn 

y_true = predictions_test.select(['label']).collect()
y_pred = predictions_test.select(['prediction']).collect()

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_true, y_pred))


# In[ ]:


from handyspark import *
predictionFinal1 = predictionFinal.toHandy().cols[['probability', 'prediction', 'Label']][:5]


# In[ ]:


from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Let's use the run-of-the-mill evaluator
evaluator = BinaryClassificationEvaluator(labelCol='Label')

# We have only two choices: area under ROC and PR curves :-(
auroc = evaluator.evaluate(predictionFinal, {evaluator.metricName: "areaUnderROC"})
auprc = evaluator.evaluate(predictionFinal, {evaluator.metricName: "areaUnderPR"})
print("Area under ROC Curve: {:.4f}".format(auroc))
print("Area under PR Curve: {:.4f}".format(auprc))


# In[ ]:


from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Let's use the run-of-the-mill evaluator
evaluator = BinaryClassificationEvaluator(labelCol='Label')

# We have only two choices: area under ROC and PR curves :-(
auroc = evaluator.evaluate(predictionFinaltr, {evaluator.metricName: "areaUnderROC"})
auprc = evaluator.evaluate(predictionFinaltr, {evaluator.metricName: "areaUnderPR"})
print("Area under ROC Curve: {:.4f}".format(auroc))
print("Area under PR Curve: {:.4f}".format(auprc))


# In[ ]:


from pyspark.mllib.evaluation import BinaryClassificationMetrics
from matplotlib import pyplot as plt
bcm = BinaryClassificationMetrics(predictionFinal, scoreCol='probability', labelCol='Label')

# We still can get the same metrics as the evaluator...
print("Area under ROC Curve: {:.4f}".format(bcm.areaUnderROC))
print("Area under PR Curve: {:.4f}".format(bcm.areaUnderPR))

# But now we can PLOT both ROC and PR curves!
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
bcm.plot_roc_curve(ax=axs[0])
bcm.plot_pr_curve(ax=axs[1])


# In[ ]:


from pyspark.mllib.evaluation import BinaryClassificationMetrics
from matplotlib import pyplot as plt
bcm = BinaryClassificationMetrics(predictionFinaltr, scoreCol='probability', labelCol='Label')

# We still can get the same metrics as the evaluator...
print("Area under ROC Curve: {:.4f}".format(bcm.areaUnderROC))
print("Area under PR Curve: {:.4f}".format(bcm.areaUnderPR))

# But now we can PLOT both ROC and PR curves!
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
bcm.plot_roc_curve(ax=axs[0])
bcm.plot_pr_curve(ax=axs[1])


# In[ ]:


prediction = model2.transform(numericTest)
predictionFinal_rfc = prediction.select(
    "MeaningfulWords", "prediction", "Label", "probability")
predictionFinal_rfc.show(n=4, truncate = False)
correctPrediction_rfc = predictionFinal_rfc.filter(
    predictionFinal_rfc['prediction'] == predictionFinal_rfc['Label']).count()
totalData = predictionFinal_rfc.count()
print("correct prediction:", correctPrediction_rfc, ", total data:", totalData, 
      ", accuracy:", correctPrediction_rfc/totalData)


# In[ ]:


import sklearn 

y_true = predictions_test_rfc.select(['label']).collect()
y_pred = predictions_test_rfc.select(['prediction']).collect()

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_true, y_pred))


# In[ ]:


predictionFinal2 = predictionFinal_rfc.toHandy().cols[['probability', 'prediction', 'Label']][:5]


# In[ ]:


from pyspark.ml.evaluation import BinaryClassificationEvaluator
# Let's use the run-of-the-mill evaluator
evaluator = BinaryClassificationEvaluator(labelCol='Label')

# We have only two choices: area under ROC and PR curves :-(
auroc = evaluator.evaluate(predictionFinal2, {evaluator.metricName: "areaUnderROC"})
auprc = evaluator.evaluate(predictionFinal2, {evaluator.metricName: "areaUnderPR"})
print("Area under ROC Curve: {:.4f}".format(auroc))
print("Area under PR Curve: {:.4f}".format(auprc))


# In[ ]:


prediction = model3.transform(numericTest)
predictionFinal = prediction.select(
    "MeaningfulWords", "prediction", "Label", "probability")
predictionFinal.show(n=4, truncate = False)
correctPrediction = predictionFinal.filter(
    predictionFinal['prediction'] == predictionFinal['Label']).count()
totalData = predictionFinal.count()
print("correct prediction:", correctPrediction, ", total data:", totalData, 
      ", accuracy:", correctPrediction/totalData)


# In[ ]:


predictions_test_fm = model4.transform(numericTest)


# In[ ]:


prediction = model4.transform(numericTest)
predictionFinal_rfc = prediction.select(
    "MeaningfulWords", "prediction", "Label", "probability")
predictionFinal_rfc.show(n=4, truncate = False)
correctPrediction_rfc = predictionFinal_rfc.filter(
    predictionFinal_rfc['prediction'] == predictionFinal_rfc['Label']).count()
totalData = predictionFinal_rfc.count()
print("correct prediction:", correctPrediction_rfc, ", total data:", totalData, 
      ", accuracy:", correctPrediction_rfc/totalData)

