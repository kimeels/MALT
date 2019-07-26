from sklearn.ensemble import RandomForestClassifier

class RFclassifier(RandomForestClassifier):
     def __init__(self, n_estimators='warn', criterion='gini'):
        super(RFclassifier,self).__init__(n_estimators=n_estimators,criterion=criterion)
