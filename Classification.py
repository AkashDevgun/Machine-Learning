import graphlab
from graphlab import SFrame
from graphlab.toolkits.feature_engineering import NumericImputer, CategoricalImputer
from collections import Counter
import pandas as pd

from sklearn.feature_extraction import DictVectorizer




# songs = SFrame.read_csv('song_data.csv')

# print songs[str]

# print "Next"

# print songs[int]




ratings = SFrame.read_csv('train.csv')

print ratings



#ratings['nwsentence'] = graphlab.text_analytics.tokenize(ratings['sentence'], delimiters=None)

#print ratings['nwsentence'][0]

#print A.shape

ratings['nwsentence'] = graphlab.text_analytics.count_words(ratings['sentence'], delimiters=None)

print ratings['nwsentence'][0]

ratings['newsentence'] = ratings['nwsentence'].dict_trim_by_keys(graphlab.text_analytics.stopwords(), exclude=True)


print ratings['newsentence'][0]


spoilerR = ratings[ratings['spoiler'] == "True"]

spoilerNR = ratings[ratings['spoiler'] == "False"]

dictSp = {}

dictNSp = {}  

for sentence in spoilerR['newsentence']:
	for word in sentence:
		dictSp[word]  =  dictSp.get(word, 0) + 1



for sentence in spoilerNR['newsentence']:
	for word in sentence:
		dictNSp[word]  =  dictNSp.get(word, 0) + 1
		
	
# SortedSp = sorted(dictSp, key=d.get, reverse=True)
# SortedNSp = sorted(dictSp, key=d.get, reverse=True)

listSp =  dict(Counter(dictSp).most_common(50))

listNSp =  dict(Counter(dictNSp).most_common(50))

words = []

for word, count in listSp.iteritems():
	if count > 150 and listNSp.get(word, 0) > 100:
		words.append((word, count, listNSp.get(word, 0), float(count) / float(listNSp.get(word, 0))))


# print words
# print A.shape



# print ratings['nwsentence'][0]
# print ratings['nwsentence'][2]

#ratings['sentence'] = ratings['sentence'].dict_trim_by_keys(graphlab.text_analytics.stopwords(), exclude=True)

ratings['is_spoiler'] = ratings['spoiler'] == "True"

print Counter(ratings['is_spoiler'])


ratings['tfidf'] = graphlab.text_analytics.tf_idf(ratings['newsentence'])


ratings['Feature1'] = ratings['sentence'].apply(lambda x: 'True' if '.' in x else 'False')

ratings['Feature2'] = ratings['sentence'].apply(lambda x: 'True' if ',' in x else 'False')

# ratings['Feature3'] = ratings['sentence'].apply(lambda x: 'True' if 'killed' in x else 'False')

# ratings['Feature4'] = ratings['sentence'].apply(lambda x: 'True' if 'kill' in x else 'False')

# ratings['Feature5'] = ratings['sentence'].apply(lambda x: 'True' if 'killing' in x else 'False')

ratings['Feature6'] = ratings['sentence'].apply(lambda x: 'True' if "'s" in x else 'False')

ratings['Feature7'] = ratings['sentence'].apply(lambda x: 'True' if 'season' in x else 'False')



# ratings['Feature1'] = ratings['sentence'].apply(lambda x: x.count('.'))

# ratings['Feature2'] = ratings['sentence'].apply(lambda x: x.count(','))

# ratings['Feature3'] = ratings['sentence'].apply(lambda x: x.count('killed'))

# ratings['Feature4'] = ratings['sentence'].apply(lambda x: x.count('kill'))

# ratings['Feature5'] = ratings['sentence'].apply(lambda x: x.count('killing'))



ratings = ratings.pack_columns(column_prefix='Feature', dtype=list, new_column_name='allfeatures')

#ratings['comma'] = ratings['sentence'].apply(lambda x: 'True' if '.' in x else 'False')

cols = ['verb', 'page']

checking = pd.get_dummies(ratings[cols].to_dataframe())

final = SFrame(data = checking)

fincols = final.column_names()

fincols.append('tfidf')

ratings.add_columns(final)

print ratings.shape


train, test = ratings.random_split(0.7, seed = 5)

# train['num_tags'] = train['tfidf'].apply(lambda x: len(x))

# test['num_tags'] = test['tfidf'].apply(lambda x: len(x))


# train['length'] = train['sentence'].apply(lambda x: len(x))

# test['length'] = test['sentence'].apply(lambda x: len(x))

# print test['sentence'][0]

# print test['length'][0]

#print A.shape

# train['count'] = train['sentence'].count_words()

# test['count'] = test['sentence'].count_words()


# print test['count'][0]

# print A.shape

# folds = graphlab.cross_validation.KFold(ratings, 5)
# for train, valid in folds:
#         m = graphlab.logistic_classifier.create(train, target = 'is_spoiler', features = ['tfidf'], max_iterations= 2000, l2_penalty=0.0, l1_penalty= 1.0, convergence_threshold = 0.001, solver = 'fista')
#         print m.evaluate(valid)



# for train, valid in folds:
#         m =  graphlab.boosted_trees_classifier.create(train, target='is_spoiler', features = ['tfidf'])
#         print m.evaluate(valid)

# print A.shape


model = graphlab.logistic_classifier.create(train, target = 'is_spoiler', features = ['nwsentence', 'newsentence', 'allfeatures'], max_iterations= 2000, l2_penalty=0.0, l1_penalty= 1.0, convergence_threshold = 0.005, solver = 'fista')


print model.evaluate(test)


model = graphlab.svm_classifier.create(train, target = 'is_spoiler', features = ['nwsentence', 'newsentence', 'allfeatures'], max_iterations= 200, penalty= 5, convergence_threshold = 0.005)


print model.evaluate(test)


print '###################################################'

print A.shape

model = graphlab.boosted_trees_classifier.create(train, target='is_spoiler', features = ['tfidf'])

print model.evaluate(test)

print A.shape

print ratings['tfidf'][0].values()

print ratings['tfidf'][0].keys()


print ratings


encoder = graphlab.feature_engineering.TFIDF('sentence')

transformed_sf = encoder.fit_transform(dict(ratings['sentence']))


print bogW[0].keys()[:5]

# uniquestats = ratings['OpenStatus'].unique()

# print uniquestats

# ratings['isClosed'] = ratings['OpenStatus'] == 'closed'

# print Counter(ratings['isClosed'])


print A.shape

# numericdata = ratings[float]


# numericCols = numericdata.column_names()

# stringdata = ratings[str]

# print stringdata.column_names()


# elm,num =  Counter(stringdata['f_61']).most_common(1)[0]

# print elm
# print num

# imputer = graphlab.feature_engineering.NumericImputer(strategy = 'mean')

# trans_num = imputer.fit_transform(numericdata)

# trans_num.add_columns(stringdata)


# for each in stringdata.column_names():
# 	cat_imputer = graphlab.feature_engineering.CategoricalImputer(feature = each, reference_features = numericCols)
# 	trans_num = cat_imputer.fit_transform(trans_num)


#print trans_num.column_names()



# cols = ['predicted_feature_f_61', 'predicted_feature_f_121', 'predicted_feature_f_215', 'predicted_feature_f_237']
# probcols = ['feature_probability_f_61', 'feature_probability_f_121', 'feature_probability_f_215', 'feature_probability_f_237']


# df = trans_num.to_dataframe()

# vec = DictVectorizer()

# vec_data = pd.DataFrame(vec.fit_transform(df[cols].to_dict(outtype='records')).toarray())
# vec_data.columns = vec.get_feature_names()
# vec_data.index = df.index



# print numericdata['f_5']
# print trans_num['f_5']

# print trans_num['predicted_feature_f_61']
# print stringdata['f_61']

# print trans_num['predicted_feature_f_237']
# print stringdata['f_237']

encoded = SFrame(data = vec_data)

checking = pd.get_dummies(df[cols])

print encoded.shape
print checking.shape

print trans_num.shape

final = SFrame(data = checking)
trans_num.remove_columns(cols)
trans_num.remove_columns(probcols)
trans_num.remove_columns(stringdata.column_names())
print trans_num.shape
trans_num.add_columns(final)

print trans_num.shape



train, test = trans_num.random_split(0.8, seed = 5)
model = graphlab.random_forest_regression.create(train, target='target', random_seed = 0, max_iterations = 200, max_depth = 10, row_subsample = 0.63, column_subsample = 0.33, verbose = True, metric = 'rmse', validation_set = test)

predictions = model.predict(test)
results = model.evaluate(test, 'rmse')

print results

print model.get_feature_importance()

savedRF = model.get_feature_importance()

featureRF = savedRF[savedRF['count'] > 500]['name']


print featureRF

prunedRF = trans_num.select_columns(featureRF)
prunedRF.add_column(trans_num['target'], name = 'target')

train, test = prunedRF.random_split(0.8, seed = 6)
modelPRF = graphlab.random_forest_regression.create(train, target='target', random_seed = 0, max_iterations = 200, max_depth = 10, row_subsample = 0.63, column_subsample = 1, verbose = True, metric = 'rmse', validation_set = test)

#modelPRF = graphlab.boosted_trees_regression.create(train, target='target', max_iterations = 2000, random_seed = 7, verbose = True, metric = 'rmse', validation_set = test, max_depth = 4)

predictions = modelPRF.predict(test)
results = modelPRF.evaluate(test, 'rmse')

print results

print A.shape


train, test = trans_num.random_split(0.8, seed = 5)
modelboost = graphlab.boosted_trees_regression.create(train, target='target', max_iterations = 350, random_seed = 0, verbose = True, metric = 'rmse', validation_set = test, max_depth = 5)

resultsboost = modelboost.evaluate(test, 'rmse')

print resultsboost

print modelboost.get_feature_importance()

savedBoost = modelboost.get_feature_importance()


featureboost = savedBoost[savedBoost['count'] > 30]['name']

print savedBoost['count']

print featureboost


featEx = modelboost.extract_features(test)

# featEx = trans_num['regression_tree_features']

print featEx[0]

print featEx[1]

print A.shape

prunedboost = trans_num.select_columns(featureboost)
prunedboost.add_column(trans_num['target'], name = 'target')

train, test = prunedboost.random_split(0.8, seed = 5)
modelboost = graphlab.boosted_trees_regression.create(train, target='target', max_iterations = 700, random_seed = 0, verbose = True, metric = 'rmse', validation_set = test, max_depth = 3)

resultsboost = modelboost.evaluate(test, 'rmse')

print resultsboost