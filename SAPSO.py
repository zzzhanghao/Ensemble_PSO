
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn import preprocessing

seed = 1603
np.random.seed(seed)

class getData:
	def __init__(self, filename):
		self.data = pd.read_csv(filename, index_col = 0)

	def getY(self):
		group = self.data['Group'].values
		le = preprocessing.LabelEncoder()
		y=le.fit_transform(group)
		return y
	def getX(self):
		x = self.data.drop('Group', axis=1).values
		return x


def errorFunction(classifier):
    classifierFit = classifier.fit(X_train, y_train)
    predictions = classifierFit.predict_proba(X_test)
    score = log_loss(y_test, predictions)
    return score
def generatePop(pop_size):
    hyperPar = []
    for i in range(pop_size):
        particle = {'RF': np.random.randint(20, 500), 'LDA': np.random.uniform(0.001, 0.00001),
                    'SVM': np.random.uniform(0.1, 7.0),
                    'LogReg': [np.random.uniform(0.1, 7.0), np.random.randint(10, 1000)],
                    'KNN': np.random.randint(10, 100)}  # hyperparam
        hyperPar.append(particle)
    swarm = []
    for i in range(pop_size):
        p = {}
        p[0] = hyperPar[i]  # hyperparameters
        p[1] = 999999999  # logLoss and roc_auc
        p[2] = 0.0  # velocity
        p[3] = p[0]  # best
        swarm.append(p)
    return swarm
# Load  dataset




def PSO(message, imp, c1, c2, w, max_iter, swarm, pop_size):
    fileBestP = message + imp + "BestParticle.txt"
    out2 = open(fileBestP, "w")

    out2.write(
        "Params:\n" + "c1: " + str(c1) + "\n" + "c2: " + str(c2) + "\n" + "w: " + str(w) + "\n" + "swarm size: " + str(
            pop_size) + "\n" + "iterations: " + str(iter_max) + "\n")
    fileSTR = message + imp + "Amazon_trace.csv"
    out = open(fileSTR, "w")
    out.write("Iteration,LogLoss\n")
    print(swarm[0])
    #### Random Start ###
    rf = RandomForestClassifier(random_state=seed)
    svm = SVC(probability=True, random_state=seed)
    lg = LogisticRegression(random_state=seed)
    knn = KNeighborsClassifier()
    lda = LinearDiscriminantAnalysis()
    vote = VotingClassifier(estimators=[('SVM', svm), ('Random Forests', rf), ('LogReg', lg), ('KNN', knn), ('LDA', lda)],
                            voting='soft')
    randomF = errorFunction(vote)
    out.write(str(0) + "," + str(randomF) + "\n")
    #### Random Start ###


    j = 0
    # let the first particle be the global best
    gbest = swarm[0]
    while j < iter_max:

        print("\n\n-----> " + str(j))
        for p in swarm:
            rf = RandomForestClassifier(n_estimators=p[0]['RF'], random_state=seed, bootstrap=True)
            svm = SVC(kernel='poly', gamma='auto', C=p[0]['SVM'], probability=True, random_state=seed)
            lg = LogisticRegression(solver='newton-cg', max_iter=p[0]['LogReg'][1], C=p[0]['LogReg'][0], random_state=seed)
            knn = KNeighborsClassifier(n_neighbors=p[0]['KNN'])
            lda = LinearDiscriminantAnalysis(solver='svd', tol=p[0]['LDA'])
            vote = VotingClassifier(
                estimators=[('SVM', svm), ('Random Forests', rf), ('LogReg', lg), ('KNN', knn), ('LDA', lda)],
                voting='soft')
            fitness = errorFunction(vote)
            # print(fitness)
            if fitness < gbest[1]:
                print('\n*** Global Best! ' + str(fitness) + "\n")
                gbest = p
                gbest[3] = p[0]
            if fitness < p[1]:
                # print("--- Local Best! "+str(fitness))
                p[1] = fitness
                p[3] = p[0]
            else:
                p[1] = fitness
            for clf in p[0].keys():
                if clf == 'LogReg':
                    ## C
                    v = w * p[2] + c1 * np.random.uniform(0, 1) * (p[3][clf][0] - p[0][clf][0]) + c2 * np.random.uniform(0,
                                                                                                                         1) * (
                                    gbest[3][clf][0] - p[3][clf][0])
                    p[0][clf][0] = abs(p[0][clf][0] + v)
                    ## max_iter
                    v = w * p[2] + c1 * np.random.uniform(0, 1) * (p[3][clf][1] - p[0][clf][1]) + c2 * np.random.uniform(0,
                                                                                                                         1) * (
                                    gbest[3][clf][1] - p[3][clf][1])
                    p[0][clf][1] = abs(p[0][clf][1] + int(v))
                elif clf == 'RF':
                    ## n_estimarors (RF)
                    v = w * p[2] + c1 * np.random.uniform(0, 1) * (p[3][clf] - p[0][clf]) + c2 * np.random.uniform(0, 1) * (
                                gbest[3][clf] - p[3][clf])
                    newPosition = abs(p[0][clf] + int(v))
                    if newPosition < 10:
                        print("Forest < 10")
                        p[0][clf] = np.random.randint(50, 500)
                    else:
                        p[0][clf] = newPosition
                elif clf == 'KNN':  # n_neighbors (KNN)
                    v = w * p[2] + c1 * np.random.uniform(0, 1) * (p[3][clf] - p[0][clf]) + c2 * np.random.uniform(0, 1) * (
                                gbest[3][clf] - p[3][clf])
                    newPosition = abs(p[0][clf] + int(v))
                    if newPosition < 5 or newPosition > 300:
                        print("N neighbours < 5")
                        p[0][clf] = np.random.randint(5, 100)
                    else:
                        p[0][clf] = newPosition


                else:  # C (SVM)
                    v = w * p[2] + c1 * np.random.uniform(0, 1) * (p[3][clf] - p[0][clf]) + c2 * np.random.uniform(0, 1) * (
                                gbest[3][clf] - p[3][clf])
                    p[0][clf] = abs(p[0][clf] + v)

        out.write(str(j + 1) + "," + str(gbest[1]) + "\n")

        j += 1
    for clf in gbest[0].keys():
        out2.write(clf + '\t' + str(gbest[0][clf]) + '\n')

print("reading data...")
MLData = getData("../../ADB_code/expData_PSO.csv")
y = MLData.getY()
X = MLData.getX()

X_Pretrain, X_validation, y_Pretrain, y_validation = train_test_split(X, y, test_size = .20, random_state=seed, stratify=y)
X_train, X_test, y_train, y_test = train_test_split(X_Pretrain, y_Pretrain, test_size = .20, random_state=seed, stratify=y_Pretrain)

iter_max = 30
c1 = 1.5
c2 = 2
w =  0.8
pop_size = 60
swarm  = generatePop(pop_size)

PSO("test", "Unbalanced",c1,c2,w,iter_max, swarm,pop_size)