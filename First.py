import graphlab
from graphlab import SFrame
from graphlab.toolkits.feature_engineering import NumericImputer, CategoricalImputer
from collections import Counter
import pandas as pd

from sklearn.feature_extraction import DictVectorizer




#songs = SFrame.read_csv('song_data.csv')

#print songs[str]

#print "Next"

#print songs[int]




ratings = SFrame.read_csv('codetest_train.txt', delimiter='\t', na_values=['NA', ''])

numericdata = ratings[float]


numericCols = numericdata.column_names()

stringdata = ratings[str]

print stringdata.column_names()


# elm,num =  Counter(stringdata['f_61']).most_common(1)[0]

# print elm
# print num

imputer = graphlab.feature_engineering.NumericImputer(strategy = 'mean')

trans_num = imputer.fit_transform(numericdata)

trans_num.add_columns(stringdata)


for each in stringdata.column_names():
	cat_imputer = graphlab.feature_engineering.CategoricalImputer(feature = each, reference_features = numericCols)
	trans_num = cat_imputer.fit_transform(trans_num)


#print trans_num.column_names()



cols = ['predicted_feature_f_61', 'predicted_feature_f_121', 'predicted_feature_f_215', 'predicted_feature_f_237']
probcols = ['feature_probability_f_61', 'feature_probability_f_121', 'feature_probability_f_215', 'feature_probability_f_237']


df = trans_num.to_dataframe()

vec = DictVectorizer()

vec_data = pd.DataFrame(vec.fit_transform(df[cols].to_dict(outtype='records')).toarray())
vec_data.columns = vec.get_feature_names()
vec_data.index = df.index



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


#model = graphlab.linear_regression.create(train, target='target', max_iterations = 200, l2_penalty=0.5, l1_penalty=0.01, validation_set = test)

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