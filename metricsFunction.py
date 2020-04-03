from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,VotingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split, StratifiedKFold 
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn import preprocessing

from sklearn.metrics import precision_recall_curve, average_precision_score, precision_recall_fscore_support, classification_report
from sklearn.metrics import roc_curve, auc, roc_auc_score, f1_score, log_loss, precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn import metrics


seed = 1603
np.random.seed(seed)


adaData = pd.read_csv("./validationDatasets/ADASYN_validationData.csv", index_col = 0)
adaData = adaData.sort_values('Group')
smoteData = pd.read_csv("./validationDatasets/SMOTE_validationData.csv", index_col = 0)
smoteData = smoteData.sort_values('Group')
expData = pd.read_csv("./validationDatasets/Original_validationData.csv", index_col = 0)
expData = expData.sort_values('Group')
print(expData.head())

rf_default = RandomForestClassifier(random_state=seed)
svm_default = SVC(probability=True, random_state=seed)
lg_default = LogisticRegression(random_state=seed)
knn_default = KNeighborsClassifier()
lda_default = LinearDiscriminantAnalysis()
vote = VotingClassifier(estimators=[('SVM', svm_default), ('Random Forests', rf_default), ('LogReg', lg_default), ('KNN', knn_default), ('LDA',lda_default)], voting='soft')

##LDA
rf_ADA = RandomForestClassifier(n_estimators = 11,random_state=seed)
svm_ADA = SVC(kernel='poly', gamma = 'auto',C =4.12481631472 ,probability=True, random_state=seed)
lg_ADA = LogisticRegression(solver = 'newton-cg',max_iter = 235,C = 135.6688, random_state=seed)
knn_ADA = KNeighborsClassifier(n_neighbors = 5)
lda_ADA = LinearDiscriminantAnalysis(tol = 0.000584259)
adasyn = VotingClassifier(estimators=[('SVM', svm_ADA), ('Random Forests', rf_ADA), ('LogReg', lg_ADA), ('KNN', knn_ADA), ('LDA',lda_ADA)], voting='soft')

rf_SMOTE = RandomForestClassifier(n_estimators = 84,random_state=seed)
svm_SMOTE = SVC(kernel='poly', gamma = 'auto',C =12.6360380346 ,probability=True, random_state=seed)
lg_SMOTE = LogisticRegression(solver = 'newton-cg',max_iter = 432,C = 50.99227570850435, random_state=seed)
knn_SMOTE = KNeighborsClassifier(n_neighbors = 5)
lda_SMOTE = LinearDiscriminantAnalysis(tol = 9.25895394348e-06)
smote = VotingClassifier(estimators=[('SVM', svm_SMOTE), ('Random Forests', rf_SMOTE), ('LogReg', lg_SMOTE), ('KNN', knn_SMOTE), ('LDA',lda_SMOTE)], voting='soft')

rf_Amazon = RandomForestClassifier(n_estimators = 389,random_state=seed)
svm_Amazon = SVC(kernel='poly', gamma = 'auto',C = 2.48906112826 ,probability=True, random_state=seed)
lg_Amazon = LogisticRegression(solver = 'newton-cg',max_iter = 1022,C = 0.0224618581186563, random_state=seed)
knn_Amazon = KNeighborsClassifier(n_neighbors = 5)
lda_Amazon = LinearDiscriminantAnalysis(tol = 0.000785350859773)
pso = VotingClassifier(estimators=[('SVM', svm_Amazon), ('Random Forests', rf_Amazon), ('LogReg', lg_Amazon), ('KNN', knn_Amazon), ('LDA',lda_Amazon)], voting='soft')
##LDA




def metrics(exp, clf, name, imp, cv):

	y = exp['Group'].values
	print(y)
	#le = preprocessing.LabelEncoder()
	# Converting string labels into numbers.
	#y=le.fit_transform(group)


	# unique, counts = np.unique(y, return_counts=True)
	# print(dict(zip(unique, counts)))

	# unique, counts = np.unique(y, return_counts=True)
	#print(dict(zip(unique, counts)))
	#yDF = expData['Group']

	X = exp.drop('Group', axis=1)

	skf = StratifiedKFold(n_splits = cv,random_state = seed)


	out = open("./validationDatasets/cvResults/"+str(cv)+"_"+str(name)+"_"+str(imp)+".csv", "w")
	print("logloss")
	logloss = cross_val_score(clf, X, y, cv=skf.split(X, y), scoring='neg_log_loss')
	print(str(logloss.mean())+"\n")
	out.write("logloss")
	for i in logloss:
		out.write(","+str(i))
	out.write("\n")

	print("accuracy")
	acc = cross_val_score(clf, X, y, cv=skf.split(X, y), scoring='accuracy')
	print(str(acc.mean())+"\n")
	out.write("accuracy")
	for i in acc:
		out.write(","+str(i))
	out.write("\n")

	print("f1")
	f1 = cross_val_score(clf, X, y, cv=skf.split(X, y), scoring='f1')
	print(str(f1.mean())+"\n")
	out.write("f1")
	for i in f1:
		out.write(","+str(i))
	out.write("\n")

	print("precision")
	precision = cross_val_score(clf, X, y, cv=skf.split(X, y), scoring='precision')
	print(str(precision.mean())+"\n")
	out.write("precision")
	for i in precision:
		out.write(","+str(i))
	out.write("\n")

	print("recall")
	recall = cross_val_score(clf, X, y, cv=skf.split(X, y), scoring='recall')
	print(str(recall.mean())+"\n")
	out.write("recall")
	for i in recall:
		out.write(","+str(i))
	out.write("\n")

	print("roc")
	roc = cross_val_score(clf, X, y, cv=skf.split(X, y), scoring='roc_auc')
	print(str(roc.mean())+"\n")
	out.write("roc_auc")
	for i in roc:
		out.write(","+str(i))
	out.write("\n")

	print("balanced accuracy")
	ba = cross_val_score(clf, X, y, cv=skf.split(X, y), scoring='balanced_accuracy')
	print(str(ba.mean())+"\n")
	out.write("balanced accuracy")
	for i in ba:
		out.write(","+str(i))
	out.write("\n")

	print("average_precision")
	ap = cross_val_score(clf, X, y, cv=skf.split(X, y), scoring='average_precision')
	print(str(ap.mean())+"\n")
	out.write("average_precision")
	for i in ap:
		out.write(","+str(i))
	out.write("\n")

	out.close()

print("ADASYN PSO")
metrics(adaData, adasyn, "LDA_ADASYN", "PSO",16)
print("ADASYN Default")
metrics(adaData, vote, "LDA_ADASYN", "default",16)
print("SMOTE PSO")
metrics(smoteData, smote, "LDA_SMOTE", "PSO",16)
print("SMOTE Default")
metrics(smoteData, vote, "LDA_SMOTE", "default",16)
print("Vote")
metrics(expData, vote, "LDA_Vote", "default",16)
print("PSO")
metrics(expData, pso, "LDA_PSO", "1603",16)






