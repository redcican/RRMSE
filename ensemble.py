from sklearn.base import BaseEstimator, TransformerMixin, clone, RegressorMixin
from sklearn.ensemble import VotingRegressor
import numpy as np

# weighted average ensemble method based on RRMSE for regression

class RRMSEModel(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    def cal_rrmse(self,actual, predict, constant):
        res = 0
        countvalue = len(actual)
        for i in range(len(actual)):
            diff = actual.iloc[i] - predict[i]
            denominator = actual.iloc[i] + constant
            res += np.power(diff/denominator,2)
        rrmse = np.sqrt(res/countvalue)

        return rrmse

    def cal_constant(self,actual):
        avg = sum(actual) / len(actual)
        sumAbsDiff = 0
        for i in actual:
            sumAbsDiff += np.absolute(i - avg)
        result = sumAbsDiff / len(actual)
    
        return result

    def _weights(self, X,y):
        error_list = []
        for model in self.models:
            model.fit(X,y)
            y_pred = model.predict(X)
            
            const = self.cal_constant(y)
            error = self.cal_rrmse(y,y_pred,const)
            error_list.append(error)

        error_list_inverse=[1/i for i in error_list]
        sum_error = sum(error_list)
        weights = [i/sum_error for i in error_list_inverse]
        
        return weights

    def fit(self, X,y):
        weights = self._weights(X,y)
        cnt = 1
        voting_model_list  = []
        for model in self.models:
            voting_model_list.append((f'R{cnt}',model))
            cnt = cnt+1
        global estimator
        estimator = VotingRegressor(voting_model_list,weights=weights)
        estimator.fit(X,y)
        return self
    
    def predict(self, X):
        return estimator.predict(X)


# bagging model for regression ensemble
class BRModel(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        for model in self.models_:
            X_tmp, y_tmp = self.subsample(X, y)
            model.fit(X_tmp, y_tmp)
        
        return self
            
    # Create a random subsample from the dataset with replacement
    def subsample(self, X, y, ratio=1.0):
        X_new, y_new = list(), list()
        n_sample = round(len(X) * ratio)
        while len(X_new) < n_sample:
            index = np.random.randint(len(X))
            X_new.append(X.iloc[index])
            y_new.append(y.iloc[index])
        return np.asarray(X_new), np.asarray(y_new)
    
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X.values) for model in self.models_
        ])
        return np.mean(predictions, axis=1)


# Majority voting ensemble for regression 
class VRUModel(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)
            
        global max_column
        max_column = self.voting(X,y)
        return self
            
    # Create a random subsample from the dataset with replacement
    def voting(self, X, y):
        # train the model
        y_train_list = []
        for model in self.models:
            model.fit(X,y)
            # make prediction on training
            y_train_pred = model.predict(X)
            y_train_list.append(y_train_pred)
            
        # transform to numpy so we can benefit from broadcast operations
        # stack the predictions horizontally so we can compare them using majority voting
        stack_pred = np.column_stack(y_train_list).T
            
        epsilon = 0.05
        acc_index = []
            
        for i,pred in enumerate(stack_pred):
            acc = np.zeros_like(pred)
            # the left neighbors
            left = pred - pred * epsilon
            # the right neighbors
            right = pred + pred * epsilon
            # construct a temp list to iterate the rest of items
            stack_temp = np.delete(stack_pred,i,0)
            for pred_sub in stack_temp:
                for j, _ in enumerate(pred_sub):
                    if pred_sub[j] >= left[j] and pred_sub[j] <= right[j]:
                        acc[j] += 1
            acc_index.append(acc.tolist())
        
        # find the max value index along columns
        # global max_column
        max_column = np.argmax(np.array(acc_index), axis=0)
        return max_column
    
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_])
        # transpose prediction list for vectorize.
        #stack_pred_t = predictions.T
        # mapping the index to their corresponding values in each rows
        final_pred = []
        for i,row in enumerate(predictions):
            value = row[max_column[i]]
            final_pred.append(value)
        return np.asarray(final_pred)
    
# Dynamic Weighting Voting Regressor
class DWRModel(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)
            
        global max_column
        max_column = self.voting(X,y)
        return self
            
    # Create a random subsample from the dataset with replacement
    def voting(self, X, y):
        # train the model
        y_train_list = []
        for model in self.models:
            model.fit(X,y)
            # make prediction on training
            y_train_pred = model.predict(X)
            y_train_list.append(y_train_pred)
            
        # transform to numpy so we can benefit from broadcast operations
        # stack the predictions horizontally so we can compare them using majority voting
        stack_pred = np.column_stack(y_train_list).T
            
        epsilon = 0.05
        acc_index = []
            
        for i,pred in enumerate(stack_pred):
            acc = np.zeros_like(pred)
            # the left neighbors
            left = pred - pred * epsilon
            # the right neighbors
            right = pred + pred * epsilon
            # construct a temp list to iterate the rest of items
            stack_temp = np.delete(stack_pred,i,0)
            for pred_sub in stack_temp:
                for j, _ in enumerate(pred_sub):
                    if pred_sub[j] >= left[j] and pred_sub[j] <= right[j]:
                        acc[j] += 1
            acc_index.append(acc.tolist())
        
        # find the max value index along columns
        # global max_column
        max_column = np.argmax(np.array(acc_index), axis=0)
        return max_column
    
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_])
        # transpose prediction list for vectorize.
        #stack_pred_t = predictions.T
        # mapping the index to their corresponding values in each rows
        final_pred = []
        for i,row in enumerate(predictions):
            value = row[max_column[i]]
            final_pred.append(value)
        return np.asarray(final_pred)
