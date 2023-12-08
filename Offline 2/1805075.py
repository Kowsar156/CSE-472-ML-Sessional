import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif

pd.options.mode.chained_assignment = None  # default='warn'
np.random.seed(1)

churnDataFileName = 'Churn\WA_Fn-UseC_-Telco-Customer-Churn.csv'
adultDataFileName_train = 'Adult\adult.data'
adultDataFileName_test = 'Adult\adult.test'
creditCardDataFileName = 'CreditCard\creditcard.csv'

def preprocessAdultDataset(numOfFeatures, testSize=0.2):

    # Step Minus One: Load dataset
    colNames = ["age","workclass","fnlwgt","education","education-num",
                "marital-status","occupation","relationship","race","sex",
                "capital-gain","capital-loss","hours-per-week","native-country","class"]
    train_data = pd.read_csv(adultDataFileName_train, header=None, names=colNames)
    test_data = pd.read_csv(adultDataFileName_test, header=None, names=colNames)

    # Step 0: modify values in some columns
    train_data["class"].replace({' >50K': 1, ' <=50K': 0}, inplace=True)
    test_data["class"].replace({' >50K.': 1, ' <=50K.': 0}, inplace=True)

    # Step 0.5: Concatenating train and test data
    train_data["is_train"] = 1
    test_data["is_train"] = 0
    data = pd.concat([train_data, test_data], ignore_index=True)

    # Step 1: Handle missing values with mean
    noneHandler = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

    for colName in ['workclass', 'occupation', 'native-country']:
        data[colName].replace({' ?': np.nan}, inplace=True)
        data[colName] = noneHandler.fit_transform(data[[colName]])
    

    # Step 2: Convert categories into one hot encoding
    data = pd.get_dummies(data, columns=["workclass","education","marital-status",
                                         "occupation","relationship","race","sex","native-country"])
    
    # Step 2.5: Separate train and test data
    train_data = data[data["is_train"] == 1]
    test_data = data[data["is_train"] == 0]

    # Step 2.75: Drop unnecessary columns
    train_data.drop(['is_train'], inplace=True, axis=1)
    test_data.drop(['is_train'], inplace=True, axis=1)


    # Step 3: Standardize numerical columns
    numColumns = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    for column in numColumns:
        train_data[column] = (data[column] - data[column].mean()) / data[column].std()
        test_data[column] = (data[column] - data[column].mean()) / data[column].std()
    

    # Step 4: Split dataset into target and features
    y_train = train_data['class']
    X_train = train_data.drop(columns=['class'])
    y_test = test_data['class']
    X_test = test_data.drop(columns=['class'])

    # Step 5: Feature selection using information gain
    if numOfFeatures != -1 and numOfFeatures < X_train.shape[1]:
        selector = SelectKBest(score_func=mutual_info_classif, k=numOfFeatures)
        X_train = selector.fit_transform(X_train, y_train)
        X_test = selector.transform(X_test)

    return X_train, X_test, y_train.values, y_test.values




def preprocessChurnDataset(numOfFeatures, testSize=0.2):

    # Step Minus One: Load dataset
    data = pd.read_csv(churnDataFileName)

    # Step 0: Drop unnecessary customerID column
    data.drop(['customerID'], inplace=True, axis=1)

    # Step 1: Handle missing values with mean
    noneHandler = SimpleImputer(missing_values=np.nan, strategy='mean')
    
    data["TotalCharges"].replace({' ': np.nan}, inplace=True)
    data["TotalCharges"] = noneHandler.fit_transform(data[["TotalCharges"]])

    #Step 1.5: Handle ultapalta values
    data["MultipleLines"].replace({'No phone service': 'No'}, inplace=True)
    data["OnlineSecurity"].replace({'No internet service': 'No'}, inplace=True)
    data["OnlineBackup"].replace({'No internet service': 'No'}, inplace=True)
    data["DeviceProtection"].replace({'No internet service': 'No'}, inplace=True)
    data["TechSupport"].replace({'No internet service': 'No'}, inplace=True)
    data["StreamingTV"].replace({'No internet service': 'No'}, inplace=True)
    data["StreamingMovies"].replace({'No internet service': 'No'}, inplace=True)
    data["Churn"].replace({'Yes': 1, 'No': 0}, inplace=True)

    # Step 2: Convert strings and categories into one hot encoding
    data = pd.get_dummies(data, columns=['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                                         'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                                         'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 
                                         'PaymentMethod'])
    
    # Step 2.5: Standardize numerical columns
    numColumns = ['tenure', 'MonthlyCharges', 'TotalCharges']
    for column in numColumns:
        data[column] = (data[column] - data[column].mean()) / data[column].std()
    

    # Step 3: Split dataset into test and train
    y = data['Churn']
    X = data.drop(columns=['Churn'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state=75)

    # Step 4: Feature selection using information gain
    if numOfFeatures != -1 and numOfFeatures < X_train.shape[1]:
        selector = SelectKBest(score_func=mutual_info_classif, k=numOfFeatures)
        X_train = selector.fit_transform(X_train, y_train)
        X_test = selector.transform(X_test)

    return X_train, X_test, y_train.values, y_test.values





def preprocessCreditCardDataset(numOfFeatures, testSize=0.2):

    # Step Minus One: Load dataset
    data = pd.read_csv(creditCardDataFileName)

    # Step 0: Filter samples with class 1 and class 0
    filtered_ones = data[data['Class'] == 1]
    filtered_zeros = data[data['Class'] == 0]

    # Step 1: Sample 20000 samples from class 0, and merge with class 1
    filtered_zeros = filtered_zeros.sample(n=20000, random_state=75)
    data = pd.concat([filtered_ones, filtered_zeros])
    
    # Step 2: Standardize numerical columns
    numColumns = [col for col in data.columns if col != 'Class']
    for column in numColumns:
        data[column] = (data[column] - data[column].mean()) / data[column].std()
    

    # Step 3: Split dataset into test and train
    y = data['Class']
    X = data.drop(columns=['Class'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state=75)

    # Step 4: Feature selection using information gain
    if numOfFeatures != -1 and numOfFeatures < X_train.shape[1]:
        selector = SelectKBest(score_func=mutual_info_classif, k=numOfFeatures)
        X_train = selector.fit_transform(X_train, y_train)
        X_test = selector.transform(X_test)

    return X_train, X_test, y_train.values, y_test.values


def evaluatePerformance(y, yP):
    TP = np.sum((y == 1) & (yP == 1))
    FN = np.sum((y == 1) & (yP == 0))
    TN = np.sum((y == 0) & (yP == 0))
    FP = np.sum((y == 0) & (yP == 1))

    accuracy = round((TP + TN)/(TP + TN + FP + FN), 4)
    #accuracy = round(np.mean(y == yP), 2)
    sensitivity = round(TP / (TP + FN), 4)
    specificity = round(TN / (TN + FP), 4)
    precision = round(TP / (TP + FP), 4)
    recall = round(TP / (TP + FN), 4)
    f1 = round(2 * precision * recall / (precision + recall), 4)
    fdr = round(FP / (FP + TP), 4)
    
    return accuracy, sensitivity, specificity, precision, recall, f1, fdr



#Logistic Regression Implementation from scratch
class LogisticRegression:
    def __init__(self, iterations, learningRate=0.01, threshold=0):
        self.learningRate = learningRate
        self.iterations = iterations
        self.threshold = threshold
    
    def sigmoid(self, x):
        x = np.array(x, dtype=np.float64)
        return (np.exp(x) / (1 + np.exp(x)))
    
    def getLoss(self, h, y):
        return np.mean((h - y)**2)
    
    def train(self, X, y):
        # weights initialization
        self.w = np.zeros(X.shape[1])
        
        for i in range(self.iterations):
            z = np.dot(X, self.w)
            h = self.sigmoid(z)
            gradient = (np.dot(X.T, (h - y)) / y.size)
            self.w = self.w - (self.learningRate * gradient)
            loss = self.getLoss(h,y)
            if loss <= self.threshold:
                break

        return self.w
    
    def predict(self, X):
        yP = self.sigmoid(np.dot(X, self.w))
        predictions = [1 if yPx >= 0.5 else 0 for yPx in yP]
        return np.array(predictions)
    

#AdaBoost    
def sigmoid(x):
    x = np.array(x, dtype=np.float64)
    return (np.exp(x) / (1 + np.exp(x)))


def Weighted_Majority_Predict(X, h, z):
    N = X.shape[0]
    K = len(h)
    yPs = []
    for k in range(K):
        yP = sigmoid(np.dot(X, h[k]))
        yPupdated = [1 if yPx >= 0.5 else -1 for yPx in yP]
        yPs.append(yPupdated)
    yPs = np.array(yPs)
    wmh = np.dot(z, yPs)
    predictions = [1 if wmh[i] >= 0 else 0 for i in range(N)]
    return np.array(predictions)


def AdaBoostPredict(X_train, x_test, y_train, K):
    Lweak = LogisticRegression(iterations=1000, learningRate=0.01, threshold=0.01)
    N = X_train.shape[0] #number of examples
    w = np.ones(N) / N
    h = []
    z = np.zeros(K)
    y_train = y_train.reshape(-1, 1)
    
    for k in range(K):
        examples = np.concatenate((X_train, y_train), axis=1)
        data = examples[np.random.choice(N, size=N, replace=True, p=w)] #Re-sampling
        
        data_X = data[:, :-1]
        data_y = data[:, -1]


        wTemp = Lweak.train(data_X, data_y)
        h.append(wTemp)
        yP = Lweak.predict(X_train)

        #accuracy = round(np.mean(y_train == yP), 2)

        error = 0
        
        for j in range(N):
            if yP[j] != y_train[j]:
                error += w[j]
        
        if error > 0.5:
            continue
        
        for j in range(N):
            if yP[j] == y_train[j]:
                w[j] *= error / (1 - error)
        
        w /= np.sum(w)
        z[k] = np.log((1 - error) / error)
    
    return Weighted_Majority_Predict(x_test, h, z)


if __name__ == "__main__":

    nFeatures = -1 # -1 for all features

    print("\n\n\nTelco Customer Churn Dataset\n\n")
    X_train, X_test, y_train, y_test = preprocessChurnDataset(numOfFeatures=nFeatures) #numOfFeatures=-1 for all features

    model = LogisticRegression(iterations=1000, learningRate=0.01, threshold=0.01)
    model.train(np.array(X_train, dtype=np.float64), y_train)

    yP = model.predict(np.array(X_train, dtype=np.float64))
    accuracy, sensitivity, specificity, precision, recall, f1, fdr = evaluatePerformance(y_train, yP)
    print("\n")
    print("Logistic Regression Results on Training Set")
    print("Accuracy: ", accuracy)
    print("True Positive Rate: ", sensitivity)
    print("True Negative Rate: ", specificity)
    print("Positive Predictive Value: ", precision)
    print("False Discover Rate: ", fdr)
    print("F1 Score: ", f1)

    yP = model.predict(np.array(X_test, dtype=np.float64))
    accuracy, sensitivity, specificity, precision, recall, f1, fdr = evaluatePerformance(y_test, yP)
    print("\n")
    print("Logistic Regression Results on Test Set")
    print("Accuracy: ", accuracy)
    print("True Positive Rate: ", sensitivity)
    print("True Negative Rate: ", specificity)
    print("Positive Predictive Value: ", precision)
    print("False Discover Rate: ", fdr)
    print("F1 Score: ", f1)


    X_train, X_test, y_train, y_test = preprocessChurnDataset(numOfFeatures=nFeatures) #numOfFeatures=-1 for all features

    yP = AdaBoostPredict(X_train, X_train, y_train, 5)
    accuracy, sensitivity, specificity, precision, recall, f1, fdr = evaluatePerformance(y_train, yP)
    print("\n")
    print("AdaBoost Result for Training Set where K = 5")
    print("Accuracy: ", accuracy)

    yP = AdaBoostPredict(X_train, X_test, y_train, 5)
    accuracy, sensitivity, specificity, precision, recall, f1, fdr = evaluatePerformance(y_test, yP)
    print("\n")
    print("AdaBoost Result for Test Set where K = 5")
    print("Accuracy: ", accuracy)



    X_train, X_test, y_train, y_test = preprocessChurnDataset(numOfFeatures=nFeatures) #numOfFeatures=-1 for all features

    yP = AdaBoostPredict(X_train, X_train, y_train, 10)
    accuracy, sensitivity, specificity, precision, recall, f1, fdr = evaluatePerformance(y_train, yP)
    print("\n")
    print("AdaBoost Result for Training Set where K = 10")
    print("Accuracy: ", accuracy)


    yP = AdaBoostPredict(X_train, X_test, y_train, 10)
    accuracy, sensitivity, specificity, precision, recall, f1, fdr = evaluatePerformance(y_test, yP)
    print("\n")
    print("AdaBoost Result for Test Set where K = 10")
    print("Accuracy: ", accuracy)



    X_train, X_test, y_train, y_test = preprocessChurnDataset(numOfFeatures=nFeatures) #numOfFeatures=-1 for all features

    yP = AdaBoostPredict(X_train, X_train, y_train, 15)
    accuracy, sensitivity, specificity, precision, recall, f1, fdr = evaluatePerformance(y_train, yP)
    print("\n")
    print("AdaBoost Result for Training Set where K = 15")
    print("Accuracy: ", accuracy)


    yP = AdaBoostPredict(X_train, X_test, y_train, 15)
    accuracy, sensitivity, specificity, precision, recall, f1, fdr = evaluatePerformance(y_test, yP)
    print("\n")
    print("AdaBoost Result for Test Set where K = 15")
    print("Accuracy: ", accuracy)



    X_train, X_test, y_train, y_test = preprocessChurnDataset(numOfFeatures=nFeatures) #numOfFeatures=-1 for all features

    yP = AdaBoostPredict(X_train, X_train, y_train, 20)
    accuracy, sensitivity, specificity, precision, recall, f1, fdr = evaluatePerformance(y_train, yP)
    print("\n")
    print("AdaBoost Result for Training Set where K = 20")
    print("Accuracy: ", accuracy)


    yP = AdaBoostPredict(X_train, X_test, y_train, 20)
    accuracy, sensitivity, specificity, precision, recall, f1, fdr = evaluatePerformance(y_test, yP)
    print("\n")
    print("AdaBoost Result for Test Set where K = 20")
    print("Accuracy: ", accuracy)







    print("\n\n\nAdult Income Dataset\n\n")
    X_train, X_test, y_train, y_test = preprocessAdultDataset(numOfFeatures=nFeatures) #numOfFeatures=-1 for all features

    model = LogisticRegression(iterations=1000, learningRate=0.01, threshold=0.01)
    model.train(np.array(X_train, dtype=np.float64), y_train)

    yP = model.predict(np.array(X_train, dtype=np.float64))
    accuracy, sensitivity, specificity, precision, recall, f1, fdr = evaluatePerformance(y_train, yP)
    print("\n")
    print("Logistic Regression Results on Training Set")
    print("Accuracy: ", accuracy)
    print("True Positive Rate: ", sensitivity)
    print("True Negative Rate: ", specificity)
    print("Positive Predictive Value: ", precision)
    print("False Discover Rate: ", fdr)
    print("F1 Score: ", f1)

    yP = model.predict(np.array(X_test, dtype=np.float64))
    accuracy, sensitivity, specificity, precision, recall, f1, fdr = evaluatePerformance(y_test, yP)
    print("\n")
    print("Logistic Regression Results on Test Set")
    print("Accuracy: ", accuracy)
    print("True Positive Rate: ", sensitivity)
    print("True Negative Rate: ", specificity)
    print("Positive Predictive Value: ", precision)
    print("False Discover Rate: ", fdr)
    print("F1 Score: ", f1)


    X_train, X_test, y_train, y_test = preprocessAdultDataset(numOfFeatures=nFeatures) #numOfFeatures=-1 for all features

    yP = AdaBoostPredict(X_train, X_train, y_train, 5)
    accuracy, sensitivity, specificity, precision, recall, f1, fdr = evaluatePerformance(y_train, yP)
    print("\n")
    print("AdaBoost Result for Training Set where K = 5")
    print("Accuracy: ", accuracy)

    yP = AdaBoostPredict(X_train, X_test, y_train, 5)
    accuracy, sensitivity, specificity, precision, recall, f1, fdr = evaluatePerformance(y_test, yP)
    print("\n")
    print("AdaBoost Result for Test Set where K = 5")
    print("Accuracy: ", accuracy)



    X_train, X_test, y_train, y_test = preprocessAdultDataset(numOfFeatures=nFeatures) #numOfFeatures=-1 for all features

    yP = AdaBoostPredict(X_train, X_train, y_train, 10)
    accuracy, sensitivity, specificity, precision, recall, f1, fdr = evaluatePerformance(y_train, yP)
    print("\n")
    print("AdaBoost Result for Training Set where K = 10")
    print("Accuracy: ", accuracy)


    yP = AdaBoostPredict(X_train, X_test, y_train, 10)
    accuracy, sensitivity, specificity, precision, recall, f1, fdr = evaluatePerformance(y_test, yP)
    print("\n")
    print("AdaBoost Result for Test Set where K = 10")
    print("Accuracy: ", accuracy)



    X_train, X_test, y_train, y_test = preprocessAdultDataset(numOfFeatures=nFeatures) #numOfFeatures=-1 for all features

    yP = AdaBoostPredict(X_train, X_train, y_train, 15)
    accuracy, sensitivity, specificity, precision, recall, f1, fdr = evaluatePerformance(y_train, yP)
    print("\n")
    print("AdaBoost Result for Training Set where K = 15")
    print("Accuracy: ", accuracy)


    yP = AdaBoostPredict(X_train, X_test, y_train, 15)
    accuracy, sensitivity, specificity, precision, recall, f1, fdr = evaluatePerformance(y_test, yP)
    print("\n")
    print("AdaBoost Result for Test Set where K = 15")
    print("Accuracy: ", accuracy)



    X_train, X_test, y_train, y_test = preprocessAdultDataset(numOfFeatures=nFeatures) #numOfFeatures=-1 for all features

    yP = AdaBoostPredict(X_train, X_train, y_train, 20)
    accuracy, sensitivity, specificity, precision, recall, f1, fdr = evaluatePerformance(y_train, yP)
    print("\n")
    print("AdaBoost Result for Training Set where K = 20")
    print("Accuracy: ", accuracy)


    yP = AdaBoostPredict(X_train, X_test, y_train, 20)
    accuracy, sensitivity, specificity, precision, recall, f1, fdr = evaluatePerformance(y_test, yP)
    print("\n")
    print("AdaBoost Result for Test Set where K = 20")
    print("Accuracy: ", accuracy)







    print("\n\n\nCredit Card Fraud Detection Dataset\n\n")
    X_train, X_test, y_train, y_test = preprocessCreditCardDataset(numOfFeatures=nFeatures) #numOfFeatures=-1 for all features

    model = LogisticRegression(iterations=1000, learningRate=0.01, threshold=0.01)
    model.train(np.array(X_train, dtype=np.float64), y_train)

    yP = model.predict(np.array(X_train, dtype=np.float64))
    accuracy, sensitivity, specificity, precision, recall, f1, fdr = evaluatePerformance(y_train, yP)
    print("\n")
    print("Logistic Regression Results on Training Set")
    print("Accuracy: ", accuracy)
    print("True Positive Rate: ", sensitivity)
    print("True Negative Rate: ", specificity)
    print("Positive Predictive Value: ", precision)
    print("False Discover Rate: ", fdr)
    print("F1 Score: ", f1)

    yP = model.predict(np.array(X_test, dtype=np.float64))
    accuracy, sensitivity, specificity, precision, recall, f1, fdr = evaluatePerformance(y_test, yP)
    print("\n")
    print("Logistic Regression Results on Test Set")
    print("Accuracy: ", accuracy)
    print("True Positive Rate: ", sensitivity)
    print("True Negative Rate: ", specificity)
    print("Positive Predictive Value: ", precision)
    print("False Discover Rate: ", fdr)
    print("F1 Score: ", f1)


    X_train, X_test, y_train, y_test = preprocessCreditCardDataset(numOfFeatures=nFeatures) #numOfFeatures=-1 for all features

    yP = AdaBoostPredict(X_train, X_train, y_train, 5)
    accuracy, sensitivity, specificity, precision, recall, f1, fdr = evaluatePerformance(y_train, yP)
    print("\n")
    print("AdaBoost Result for Training Set where K = 5")
    print("Accuracy: ", accuracy)

    yP = AdaBoostPredict(X_train, X_test, y_train, 5)
    accuracy, sensitivity, specificity, precision, recall, f1, fdr = evaluatePerformance(y_test, yP)
    print("\n")
    print("AdaBoost Result for Test Set where K = 5")
    print("Accuracy: ", accuracy)



    X_train, X_test, y_train, y_test = preprocessCreditCardDataset(numOfFeatures=nFeatures) #numOfFeatures=-1 for all features

    yP = AdaBoostPredict(X_train, X_train, y_train, 10)
    accuracy, sensitivity, specificity, precision, recall, f1, fdr = evaluatePerformance(y_train, yP)
    print("\n")
    print("AdaBoost Result for Training Set where K = 10")
    print("Accuracy: ", accuracy)


    yP = AdaBoostPredict(X_train, X_test, y_train, 10)
    accuracy, sensitivity, specificity, precision, recall, f1, fdr = evaluatePerformance(y_test, yP)
    print("\n")
    print("AdaBoost Result for Test Set where K = 10")
    print("Accuracy: ", accuracy)



    X_train, X_test, y_train, y_test = preprocessCreditCardDataset(numOfFeatures=nFeatures) #numOfFeatures=-1 for all features

    yP = AdaBoostPredict(X_train, X_train, y_train, 15)
    accuracy, sensitivity, specificity, precision, recall, f1, fdr = evaluatePerformance(y_train, yP)
    print("\n")
    print("AdaBoost Result for Training Set where K = 15")
    print("Accuracy: ", accuracy)


    yP = AdaBoostPredict(X_train, X_test, y_train, 15)
    accuracy, sensitivity, specificity, precision, recall, f1, fdr = evaluatePerformance(y_test, yP)
    print("\n")
    print("AdaBoost Result for Test Set where K = 15")
    print("Accuracy: ", accuracy)



    X_train, X_test, y_train, y_test = preprocessCreditCardDataset(numOfFeatures=nFeatures) #numOfFeatures=-1 for all features

    yP = AdaBoostPredict(X_train, X_train, y_train, 20)
    accuracy, sensitivity, specificity, precision, recall, f1, fdr = evaluatePerformance(y_train, yP)
    print("\n")
    print("AdaBoost Result for Training Set where K = 20")
    print("Accuracy: ", accuracy)


    yP = AdaBoostPredict(X_train, X_test, y_train, 20)
    accuracy, sensitivity, specificity, precision, recall, f1, fdr = evaluatePerformance(y_test, yP)
    print("\n")
    print("AdaBoost Result for Test Set where K = 20")
    print("Accuracy: ", accuracy)
