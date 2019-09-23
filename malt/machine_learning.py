from sklearn.ensemble import RandomForestClassifier

class RFclassifier(RandomForestClassifier):
     def __init__(self, n_estimators='warn', criterion='gini'):

         """
             Sklearn wrapper for random forest classifier.

             Params
             -------
             self: classifier class
                 An instance of a ML classifier.
             n_estimators: int
                Number of trees to use in random forest.
             criterion: str
                Splitting criterion.
         """
         super(RFclassifier,self).__init__(n_estimators=n_estimators,criterion=criterion)
