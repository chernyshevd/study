from collections import defaultdict
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor


sns.set(style='darkgrid')


def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class Boosting:

    def __init__(
            self,
            base_model_params: dict = None,
            n_estimators: int = 10,
            learning_rate: float = 0.1,
            subsample: float = 0.3,
            early_stopping_rounds: int = None,
            plot: bool = False,
    ):
        self.base_model_class = DecisionTreeRegressor
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate
        self.subsample: float = subsample

        self.early_stopping_rounds: int = early_stopping_rounds
        if early_stopping_rounds is not None:
            self.validation_loss = np.full(self.early_stopping_rounds, np.inf)

        self.plot: bool = plot

        self.history = defaultdict(list)
        self.loss: list = []
        self.valid_loss: list = []

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: y * self.sigmoid(-y * z)
        self.loss_derivative2 = lambda y, z: y ** 2 * self.sigmoid(-y * z) * (1 - self.sigmoid(-y * z))

    def fit_new_base_model(self, x, y, predictions):
        
        s = self.loss_derivative(y, predictions)
        
        x_smpl, _, s_smpl, _ = train_test_split(x, s, train_size=self.subsample)
        
        boot_sample = np.random.randint(x_smpl.shape[0], size = x_smpl.shape[0])
        x_bt = x_smpl[boot_sample]
        s_bt = s_smpl[boot_sample]
        
        model = self.base_model_class(**self.base_model_params)
        
        model.fit(x_bt, s_bt)
        
        new_predictions = model.predict(x)
        
        gamma = self.find_optimal_gamma(y, predictions, new_predictions)
        
        self.gammas.append(gamma)
        self.models.append(model)

    def fit(self, x_train, y_train, x_valid, y_valid):
        """
        :param x_train: features array (train set)
        :param y_train: targets array (train set)
        :param x_valid: features array (validation set)
        :param y_valid: targets array (validation set)
        """
        train_predictions = np.zeros(y_train.shape[0])
        valid_predictions = np.zeros(y_valid.shape[0])
        iter_count = 0
        
        self.history["train_loss"] = []
        self.history["valid_loss"] = []
    
        for _ in range(self.n_estimators):
    
            self.fit_new_base_model(x_train, y_train, train_predictions)
            
            train_predictions = train_predictions + self.learning_rate * self.gammas[-1] * self.models[-1].predict(x_train)
            valid_predictions = valid_predictions + self.learning_rate * self.gammas[-1] * self.models[-1].predict(x_valid)
            train_current_loss = self.loss_fn(y_train, train_predictions)
            valid_current_loss = self.loss_fn(y_valid, valid_predictions)
            
            self.history["train_loss"].append(train_current_loss)
            self.history["valid_loss"].append(valid_current_loss)
            
            
            if self.early_stopping_rounds is not None and len(self.history["valid_loss"]) > 1:           
                
                if self.history["valid_loss"][-2] < valid_current_loss:

                    iter_count += 1

                else:

                    iter_count = 0

                if self.early_stopping_rounds == iter_count:

                    break
            

    
        if self.plot:
            
            plt.figure(figsize=(10, 7))
            plt.plot(np.arange(1, len(self.history["train_loss"]) + 1), self.history["train_loss"])
            plt.plot(np.arange(1, len(self.history["train_loss"]) + 1), self.history["valid_loss"])
            plt.xlabel('n_estimators')
            plt.ylabel('loss')
            plt.legend(['train', 'valid'])
            plt.title('loss~n_estimators');
            
            
            

    def predict_proba(self, x):
        
        prediction = np.zeros(x.shape[0])
        
        for gamma, model in zip(self.gammas, self.models):
            
            prediction += self.learning_rate * gamma * model.predict(x)
        
        pos_prob = self.sigmoid(prediction)
        neg_prob = 1 - pos_prob
        
        
        return np.column_stack((neg_prob, pos_prob))
            
            

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + self.learning_rate * gamma * new_predictions) for gamma in gammas]

        return gammas[np.argmin(losses)]

    def score(self, x, y):
        return score(self, x, y)

    @property
    def feature_importances_(self):
        
        feature_importances_mean = np.mean([model.feature_importances_ for model in self.models], axis = 0)
        
        feature_importances_mean = feature_importances_mean / np.sum(feature_importances_mean)
        
        return feature_importances_mean
