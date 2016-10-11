from sklearn.linear_model import RandomizedLasso
import argparse
import numpy as np 
from sklearn.svm import LinearSVC
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LassoLarsCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.svm import LinearSVC
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import PCA
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import Ridge
import imblearn
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import EasyEnsemble
from sklearn.feature_selection import VarianceThreshold
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.kernel_ridge import KernelRidge
from sklearn.externals import joblib
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.svm import SVR
import pandas as pd 
import collections
import re
from sklearn import linear_model
from sklearn.linear_model import ElasticNet
from sklearn import preprocessing
from csv import DictReader, DictWriter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import RandomizedLasso
from sklearn import ensemble
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
#from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
import sklearn
from sklearn.cross_validation import train_test_split
import csv
from sklearn.preprocessing import Imputer
from sklearn.base import TransformerMixin
from imblearn.over_sampling import SMOTE
import seaborn as sns



sns.set()

# Define some color for the plotting
almost_black = '#262626'
palette = sns.color_palette()



class DataFrameImputer(TransformerMixin):

	def __init__(self):
		"""Impute missing values.

		Columns of dtype object are imputed with the most frequent value 
		in column.

		Columns of other types are imputed with mean of column.

		"""
	def fit(self, X, y=None):

		self.fill = pd.Series([X[c].mean()
			if X[c].dtype == np.dtype('float64') or X[c].dtype == np.dtype('int64') else X[c].value_counts().index[0] for c in X],
			index=X.columns)

		return self

	def transform(self, X, y=None):
		return X.fillna(self.fill)

	def fit_transform(self,X,y=None):
		return self.fit(X,y).transform(X)




def End ():
	print "Question_1 Finish"



if __name__ == "__main__":
	print('The scikit-learn version is {}.'.format(sklearn.__version__))
	parser = argparse.ArgumentParser(description='Question_1')

	args = parser.parse_args()

	train = pd.read_csv('train.csv')
	lengthTrain = len(train)
	print lengthTrain
	test = pd.read_csv('test.csv')
	lengthTest = len(test)
	print lengthTest

	train['online'].fillna("notonline", inplace=True)
	test['online'].fillna("notonline", inplace=True)

	ratings = train['quality'].tolist()

	intvals = []

	for each in ratings:
		intvals.append(float(each))


	train.drop('quality', axis=1, inplace=True)

	allcolumns = train.columns.values
	remove = ['id','date','tags','comments','helpfulness','clarity','easiness']
	finalcols1 = []

	for each in allcolumns:
		if each not in remove:
			finalcols1.append(each)

	#finaltrain_1 = finaltrain[finalcols1].reset_index(drop=True)
	stringfeature = ['dept','forcredit','attendance','textbookuse','interest','grade','online']
	numericfeature = []

	for each in finalcols1:
		if each not in stringfeature:
			numericfeature.append(each)

	train[finalcols1].replace('N/A',np.nan,regex=True, inplace=True)
	test[finalcols1].replace('N/A',np.nan,regex=True, inplace=True)

	numericTraindf = train[numericfeature].apply(pd.to_numeric, errors='coerce')
	numericTestdf = test[numericfeature].apply(pd.to_numeric, errors='coerce')

	stringTraindf = train[stringfeature]
	stringTestdf = test[stringfeature]

	fullstringdf = pd.concat([stringTraindf,stringTestdf])
	fullnumericdf = pd.concat([numericTraindf,numericTestdf])

	finalnumeric_df = DataFrameImputer().fit_transform(fullnumericdf)


	finalstring_df = DataFrameImputer().fit_transform(fullstringdf)

	print finalnumeric_df.shape
	print finalstring_df.shape

	numericTraindfNew, numericTestdfNew = finalnumeric_df[:lengthTrain], finalnumeric_df[lengthTrain:]

	stringTraindfNew, stringTestdfNew = finalstring_df[:lengthTrain], finalstring_df[lengthTrain:]

	print finalnumeric_df.columns.values
	print finalstring_df.columns.values

	numericTraindfNew.columns = finalnumeric_df.columns.values
	numericTestdfNew.columns = finalnumeric_df.columns.values

	stringTraindfNew.columns = finalstring_df.columns.values
	stringTestdfNew.columns = finalstring_df.columns.values


	print numericTraindfNew.shape
	print stringTraindfNew.shape
	print numericTestdfNew.shape
	print stringTestdfNew.shape
	print stringTestdfNew['grade'][0]


	laEn = preprocessing.LabelEncoder()

	stringtrainlength = len(stringTraindfNew)*len(stringTraindfNew.columns.values)
	stringtestlenth = len(stringTestdfNew)*len(stringTestdfNew.columns.values)


	train_X = np.reshape(stringTraindfNew.values,(stringtrainlength, 1)) 
	test_X = np.reshape(stringTestdfNew.values,(stringtestlenth, 1))

	fullstrings = np.vstack((train_X,test_X)) 

	print train_X.shape
	print test_X.shape
	print fullstrings.shape
	LabelEncode = laEn.fit_transform(fullstrings)

	trainstring = LabelEncode[0:stringtrainlength]
	teststring = LabelEncode[stringtrainlength:]

	print len(trainstring)
	print len(teststring)

	stringtrain_X = np.reshape(trainstring,(len(stringTraindfNew),len(stringTraindfNew.columns.values))) 
	stringtest_X = np.reshape(teststring,(len(stringTestdfNew),len(stringTestdfNew.columns.values)))

	print stringtrain_X.shape
	print stringtest_X.shape


	tags = [
  		"python,tools,python",
  		"linux,tools,ubuntu",
  		"",
  		"distributedsystems,linux,networking,tools",
	]
	vec = CountVectorizer(stop_words='english')
	tfvec = TfidfVectorizer(stop_words='english', ngram_range=(1, 1), max_features = 1500)
	#data = vec.fit_transform(tags).toarray()

	train['tags'] = train['tags'].map(lambda x: str(x)[1:-1])
	test['tags'] = test['tags'].map(lambda x: str(x)[1:-1])
	# train['tags'] = train['tags'].str.strip()
	train['tags'].replace('"','',regex=True, inplace=True)
	train['tags'].replace(' ','',regex=True, inplace=True)
	train['comments'].fillna("", inplace=True)
	test['comments'].fillna("", inplace=True)

	test['tags'].replace('"','',regex=True, inplace=True)
	test['tags'].replace(' ','',regex=True, inplace=True)
	#train['tags'] = '"' + train['tags'].astype(str) + '"'

	#port = WordNetLemmatizer()
	port = PorterStemmer()

	solution = []

	for each in train['comments'].values:
		sol = " ".join([port.stem(i) for i in each.split()])
		solution.append(sol)

		print each
		print sol
		print A.shape

	traintagdata = vec.fit_transform(train['tags'].values).toarray()
	testtagdata = vec.transform(test['tags'].values).toarray()

	print traintagdata.shape
	print testtagdata.shape

	trainTFtagdata = tfvec.fit_transform(train['comments'].values).toarray()
	testTFtagdata = tfvec.transform(test['comments'].values).toarray()

	print traintagdata.shape
	print testtagdata.shape

	print trainTFtagdata.shape
	print testTFtagdata.shape

	#train['tags'] = train['tags'].apply(lambda x: ''.join(x))

	print "See"

	t1 = pd.DataFrame(stringtrain_X)
	t2 = pd.DataFrame(traintagdata)
	t3 = pd.DataFrame(numericTraindfNew)
	t4 = pd.DataFrame(trainTFtagdata)

	d1 = pd.DataFrame(stringtest_X)
	d2 = pd.DataFrame(testtagdata)
	d3 = pd.DataFrame(numericTestdfNew)
	d4 = pd.DataFrame(testTFtagdata)

	resulttrain = pd.concat([t1, t2, t3, t4], axis=1)
	resulttest = pd.concat([d1, d2, d3, d4], axis=1)

	FullTrain = resulttrain.values
	FullTest = resulttest.values

	print FullTrain.shape
	print FullTest.shape



	#print train.columns.values

	#print np.unique(train['DIM'].tolist())

	#print len(np.unique(train['Site_Yr_Mo_Dim_Att'].tolist()))

	#train1001 = train

	#print len(train1001)

	
	#print np.unique(train['SITE_NUMBER'].tolist())

	#print len(np.unique(train['SITE_NUMBER'].tolist()))

	#print collections.Counter(train['SITE_NUMBER'].tolist())


	#trainExtract = train1001[train1001.NEW_RENTALS_y >= 0]

	#trainExtracted = trainExtract[trainExtract.NEW_RENTALS_y<=12]

	#print len(trainExtracted)
	#print collections.Counter(trainExtracted['SITE_'].tolist())

	#print collections.Counter(trainExtracted['NEW_RENTALS_y'].tolist())




	#vals = np.unique(trainExtracted['Site_Yr_Mo_Dim_Att'].tolist())

	#print len(vals)

	#allcolumns = train.columns.values

	# finalcols = []

	# for eachcol in allcolumns:
	# 	uniquevals = np.unique(trainExtracted[eachcol].tolist())
	# 	if len(uniquevals) == 1:
	# 		continue 
	# 	if len(uniquevals) == 2:
	# 		if 'nan' in uniquevals:
	# 			continue
	# 		else:
	# 			finalcols.append(eachcol)
	# 	else:
	# 		finalcols.append(eachcol)

	#finaltrain = trainExtracted[allcolumns]

	#print finaltrain.columns.values
	#print len(finaltrain.columns.values)

	#remove = ['ID','SITE.NUMBER','SITE.NAME','EXPORT.DATE','DATE.RANGE','Date','state', 'city']
	#remove = ['ID','SITE_NUMBER','EXPORT.DATE','DATE.RANGE','SQ.Site.Address.1','SQ.Site.Address.2','Comp.2.Name','Comp.2.Address','Comp.3.Web.Rate', 'Comp.3.InStore.Rate', 'Date','AVG..RENT.RATE','Comp.4.Promotion']
	#finalcols1 = []

	# for each in allcolumns:
	# 	if each not in remove:
	# 		finalcols1.append(each)

	# finaltrain_1 = finaltrain[finalcols1].reset_index(drop=True)

	#print finaltrain_1.columns.values
	#print len(finaltrain_1.columns.values)

	#print finaltrain_1

	#numericfeature = []
	#stringfeature = ['DIM','ATTRIBUTE']
	#stringfeature = ['Site_Yr_Mo_Dim_Att','SITE.NAME','DIM','ATTRIBUTE','Comp.1.Promotion', 'Comp.2.Promotion']

	# for each in finaltrain_1.columns.values:
	# 	if each not in stringfeature:
	# 		numericfeature.append(each)

	# numericdf = finaltrain_1[numericfeature].apply(pd.to_numeric, errors='coerce')

	# stringdf = finaltrain_1[stringfeature]

	# dividedf = stringdf['DIM'].str.split('x').apply(pd.Series).astype(int)

	# dividedf.columns = ['A','B','C']

	# numericdf['Unit.SF'] = dividedf['A']*dividedf['B']

	# numericdf['Total.SF'] = numericdf['Unit.SF']*numericdf['TOTAL.UNITS']

	# classes = numericdf['NEW_RENTALS_y'].tolist()

	# intclasses = []

	# for each in classes:
	# 	intclasses.append(int(each))

	# interclasses = []

	# zero = 0

	# one = 1

	# for each in intclasses:
	# 	if each < 7:
	# 		interclasses.append(zero)
	# 	else:
	# 		interclasses.append(one)

	# interclasses = np.array(interclasses)

	# print len(np.unique(interclasses))


	# numericdf.drop('NEW_RENTALS_y', axis=1, inplace=True)

	#print numericdf.dtypes

	#print stringdf.dtypes


	# laEn = preprocessing.LabelEncoder()


	# finalnumeric_df = DataFrameImputer().fit_transform(numericdf)

	# #print finalnumeric_df

	# finalstring_df = DataFrameImputer().fit_transform(stringdf)

	#print len(finalstring_df)

	# colls = finalnumeric_df.columns.tolist()

	# colls.append(finalstring_df.columns.tolist())

	# print finalnumeric_df.columns.values

	# print finalstring_df.columns.values
	

	# count = 1
	# EncodeTrainData = []

	# for each in finalstring_df.columns:
	# 	if count == 1:
	# 		EncodeTrainData = finalstring_df[each]
	# 		count = 2
	# 	else:
	# 		EncodeTrainData = np.column_stack((EncodeTrainData,finalstring_df[each]))



	# print EncodeTrainData.shape

	# #+1-7174345121 sm_rm4@yahoo.com


	# train_X = np.reshape(EncodeTrainData,(len(finalstring_df)*len(finalstring_df.columns.values))) 

	# trainLabelEncode = laEn.fit_transform(train_X)

	# print laEn.classes_


	# Encodedtrain_X = np.reshape(trainLabelEncode, (len(finalstring_df),len(finalstring_df.columns.values))) 


	# count = 1
	# TrainValueData = []

	# for eachfeature in finalnumeric_df.columns:
	# 	if count == 1:
	# 		TrainValueData = finalnumeric_df[eachfeature]
	# 		count = 2
	# 	else:
	# 		TrainValueData = np.column_stack((TrainValueData, finalnumeric_df[eachfeature]))

	# #print finalnumeric_df
	


	# Final_Train_X = np.column_stack((TrainValueData, Encodedtrain_X))

	# print Final_Train_X.shape

	#columns= comdataframe.columns.values

	# comnumframe = pd.DataFrame(Final_Train_X) 

	# print comnumframe.corr(method='pearson', columns = colls)

	# print A.shape



	rng = np.random.RandomState(1)

	


	X_dummytrain, X_dummytest, y_dummytrain, y_dummytest = train_test_split(FullTrain, intvals, test_size=0.4, random_state=42)


	print X_dummytrain.shape
	print len(X_dummytrain)
	print X_dummytest.shape
	print len(X_dummytest)
	#AdaBoostClassifier(DecisionTreeClassifier(max_depth=4), n_estimators=600),
	#AdaBoostClassifier(DecisionTreeClassifier(max_depth=4), n_estimators=600),

	# featureclassifiers = [
	# ensemble.RandomForestClassifier(max_depth=10, n_estimators=100),
	# DecisionTreeClassifier(max_depth=10),
	# ensemble.ExtraTreesClassifier(n_estimators = 100),
	# ensemble.GradientBoostingClassifier(n_estimators = 500, learning_rate = 0.1, max_depth = 10 , random_state = 0),
	# LinearSVC(C=0.01, penalty="l1", dual=False),
	# SVC(decision_function_shape='ovo')]
	# lsvc = Lasso(alpha = 0.05).fit(X_dummytrain,y_dummytrain)
	# model = SelectFromModel(lsvc, prefit = True)

	# Train_new = model.transform(X_dummytrain)
	# print X_dummytrain.shape
	# print Train_new.shape
	# newindices = model.get_support(True)

	# print newindices

	# FinalTrainLessFeature = X_dummytrain[np.ix_(np.arange(len(X_dummytrain)), newindices)]
	# FinalTestLessFeature = X_dummytest[np.ix_(np.arange(len(X_dummytest)), newindices)]

	# estimate = ensemble.ExtraTreesRegressor(bootstrap=True,max_depth=12,n_estimators = 300).fit(FinalTrainLessFeature,y_dummytrain)

	# predictions = estimate.predict(FinalTestLessFeature)

	# print mean_squared_error(y_dummytest, predictions)
	# df = pd.DataFrame({"Actual": y_dummytest, "Predicted": predictions})
	# df["Residue"] = df["Predicted"] - df["Actual"]
	# print "First Step"
	# print df["Residue"].mean()
	# p = 0.1
	# while p < 100:
	# 	df["Predicted"] = df["Predicted"] + p
	# 	df["Actual"] = df["Actual"] + p
	# 	df["Residue"] = df["Predicted"] - df["Actual"]
	# 	means = df["Residue"].mean() 
	# 	if abs(means) > 0.38 and abs(means) <0.42:
	# 		break;
	# 	p = p + 0.1

	# print p
	# print A.shape


	# print df.columns.values

	# for x in xrange(0,5):
	# 	print x
	# 	minidf = df[df.Actual == x]	
	# 	print minidf["Predicted"].mean()
	# 	print minidf["Predicted"].std()
	# 	print "Next"
	# plt.scatter(df['Actual'].tolist(), df['Residue'].tolist())

	# plt.show()

	# filename = 'Regressor.joblib.pkl'
	# _ = joblib.dump(estimate, filename, compress=9)

	# print A.shape
	#Lasso(alpha = 0.01),

	#ElasticNet(alpha=0.01, l1_ratio=0.7),

	featuredclassifierss = [
	linear_model.LinearRegression(),
	Ridge(alpha=0.1),
	ensemble.ExtraTreesRegressor(n_estimators = 300),
	ElasticNet(alpha=0.01, l1_ratio=0.3),
	ensemble.GradientBoostingRegressor(alpha=0.1, n_estimators = 500, learning_rate = 0.1, max_depth = 10 , random_state = 0)]

	featureclassifiers = [
	ensemble.GradientBoostingRegressor(alpha=0.1, n_estimators = 500, learning_rate = 0.1, max_depth = 10 , random_state = 0),
	linear_model.LinearRegression(),
	Ridge(alpha=0.1),
	Ridge(alpha=0.01),
	linear_model.SGDRegressor(),
	ensemble.RandomForestRegressor(max_depth=12, n_estimators=400),
	ensemble.ExtraTreesRegressor(n_estimators = 300),
	ensemble.GradientBoostingRegressor(alpha=0.1, n_estimators = 500, learning_rate = 0.1, max_depth = 15 , random_state = 0)]

	# featureclassifiers = [
	# ensemble.ExtraTreesRegressor(n_estimators = 100)]


	for clf in featuredclassifierss:
		estimate = clf.fit(X_dummytrain,y_dummytrain)
	 	predictions = estimate.predict(X_dummytest)

	 	#print accuracy_score(y_dummytest, predictions)
		print mean_squared_error(y_dummytest, predictions)

	print "Next"

	for clf in featuredclassifierss:
		lsvc = clf.fit(X_dummytrain,y_dummytrain)
		model = SelectFromModel(lsvc, prefit = True)

		print clf

		Train_new = model.transform(X_dummytrain)
		print X_dummytrain.shape
		print Train_new.shape
		newindices = model.get_support(True)

		FinalTrainLessFeature = X_dummytrain[np.ix_(np.arange(len(X_dummytrain)), newindices)]
		FinalTestLessFeature = X_dummytest[np.ix_(np.arange(len(X_dummytest)), newindices)]

		print FinalTrainLessFeature.shape
		print FinalTestLessFeature.shape
		
		#print newindices

		for cllf in featureclassifiers:
			rng = np.random.RandomState(1)

			estimate = cllf.fit(FinalTrainLessFeature,y_dummytrain)

			predictions = estimate.predict(FinalTestLessFeature)

			#print accuracy_score(y_dummytest, predictions)
			print mean_squared_error(y_dummytest, predictions)

			#df = pd.DataFrame({"Actual": y_dummytest, "Predicted": predictions})


			#print df
		

		FinalTestLessFeature = []
		FinalTrainLessFeature = []


	featuredclassifiers = [
	GridSearchCV(ensemble.RandomForestRegressor(), cv= 10, param_grid = {"bootstrap":[True,False],"max_depth":[10, 11, 12], "n_estimators":[350, 362, 375, 387, 400]}),
	GridSearchCV(ensemble.ExtraTreesRegressor(), cv= 10, param_grid = {"bootstrap":[True,False],"max_depth":[10, 11, 12], "n_estimators":[300, 350, 375, 387, 400]}),
	GridSearchCV(ensemble.GradientBoostingRegressor(random_state = 0), cv= 10, param_grid = {"alpha":[0.05, 0.1, 0.3, 0.6, 0.9],"learning_rate":[0.05, 0.1, 0.3, 0.6, 0.9],"max_depth":[10, 11, 12], "n_estimators":[300, 350, 375, 387, 450, 500]})]



	# GridSearchCV(LassoCV(max_iter=10000), cv= 10, param_grid = {"cv":[8, 10, 12, 14, 16]}),
	# GridSearchCV(Lasso(max_iter=10000), cv= 10, param_grid = {"alpha":[0.05, 0.1, 0.3, 0.6, 1.0]}),
	
	# rfR = GridSearchCV(ensemble.RandomForestRegressor(), cv= 10,
	# 					param_grid = {"max_depth":[10, 11, 12], "n_estimators":[375, 387, 400]})

	# GridSearchCV(ensemble.RandomForestRegressor(), cv= 10, param_grid = {"max_depth":[10, 11, 12], "n_estimators":[350, 362, 375, 387, 400]})

	# rfR.fit(X_dummytrain,y_dummytrain)

	# print (rfR.best_estimator_)
	# print (rfR.best_estimator_.max_depth)
	# print (rfR.best_estimator_.n_estimators)

	# print A.shape

	# for clf in featuredclassifiers:
	# 	clf.fit(X_dummytrain,y_dummytrain)
	# 	predictions = clf.predict(X_dummytest)

	# 	#print accuracy_score(y_dummytest, predictions)
	# 	print mean_squared_error(y_dummytest, predictions)

	print "Start"	

	for clf in featuredclassifiers:
		clf.fit(X_dummytrain,y_dummytrain)
		predictions = clf.predict(X_dummytest)
		print mean_squared_error(y_dummytest, predictions)
		print "Advanced"
		lsvc = clf.best_estimator_
		model = SelectFromModel(lsvc, prefit = True)

		print lsvc

		Train_new = model.transform(X_dummytrain)
		print X_dummytrain.shape
		print Train_new.shape
		newindices = model.get_support(True)

		FinalTrainLessFeature = X_dummytrain[np.ix_(np.arange(len(X_dummytrain)), newindices)]
		FinalTestLessFeature = X_dummytest[np.ix_(np.arange(len(X_dummytest)), newindices)]

		print FinalTrainLessFeature.shape
		print FinalTestLessFeature.shape
		
		print newindices

		for cllf in featuredclassifiers:
			rng = np.random.RandomState(1)

			cllf.fit(FinalTrainLessFeature,y_dummytrain)

			predictions1 = cllf.predict(FinalTestLessFeature)

			#print accuracy_score(y_dummytest, predictions)
			print cllf.best_estimator_
			print mean_squared_error(y_dummytest, predictions1)

			#df = pd.DataFrame({"Actual": y_dummytest, "Predicted": predictions})


			#print df
		

		FinalTestLessFeature = []
		FinalTrainLessFeature = []

		print "Next Round"
	
	# FinalTrainLessFeature = Final_Train_X[np.ix_(np.arange(len(train) - 1), newindices)]
	# FinalTestLessFeature = Final_Test_X[np.ix_(np.arange(len(test)- 1), newindices)]


	# predictions = estimate.predict(FinalTestLessFeature)
	# print "In writePredictions"
	# o = DictWriter(open("Predictions.csv", 'w'),["target"])
	# for y_val in predictions:
	# 	o.writerow({'target': y_val})

	End()
	



