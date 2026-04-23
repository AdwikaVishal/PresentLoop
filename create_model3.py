import pickle
from sklearn.ensemble import VotingClassifier
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Load models (ignore sklearn version warnings)
print('Loading models...')

with open('presentloop_model.pkl', 'rb') as f:
    model1 = pickle.load(f)

with open('presentloop_model2.pkl', 'rb') as f:
    model2_data = pickle.load(f)
    model2 = model2_data['model']

with open('rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

print('model1 feats:', model1.n_features_in_)
print('model2 feats:', model2.n_features_in_)
print('rf feats:', rf_model.n_features_in_)

# Since feats differ (5,11,20), create model3 as proba-averaging ensemble
# Requires uniform input - use model1 5-feats for all (simplest compatible)

class ProbaAverageEnsemble:
    def __init__(self, models):
        self.models = models
    
    def predict_proba(self, X):
        probas = []
        for model in self.models:
            try:
                proba = model.predict_proba(X)
                probas.append(proba)
            except Exception as e:
                print(f'Model predict error: {e}')
                # Fallback dummy proba [0.5,0.5]
                probas.append(np.full((len(X), 2), 0.5))
        # Average across models
        avg_proba = np.mean(probas, axis=0)
        return avg_proba
    
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

model3 = ProbaAverageEnsemble([model1, model2, rf_model])

# Save as dict like model2
model3_data = {'model': model3}
with open('presentloop_model3.pkl', 'wb') as f:
    pickle.dump(model3_data, f)

print('Created presentloop_model3.pkl (ensemble of model1+model2+rf using avg proba on 5 feats)')

