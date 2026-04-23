import pickle
print("Inspecting presentloop_model.pkl:")
with open('presentloop_model.pkl', 'rb') as f:
    model1 = pickle.load(f)
print("Type of model1:", type(model1))
print("model1 n_features_in_:", model1.n_features_in_)
if hasattr(model1, 'predict_proba'):
    print("model1 has predict_proba")
else:
    print("model1 attrs:", dir(model1)[:10] if hasattr(model1, '__dir__') else 'No dir')
if isinstance(model1, dict):
    print("model1 keys:", list(model1.keys()))
    if 'model' in model1:
        print("Type of model1['model']:", type(model1['model']))
    if 'classifier' in model1:
        print("Type of model1['classifier']:", type(model1['classifier']))

print("\nInspecting presentloop_model2.pkl:")
with open('presentloop_model2.pkl', 'rb') as f:
    model2_data = pickle.load(f)
    model2 = model2_data['model']
print("Type of model2_data:", type(model2_data))
print("Type of model2:", type(model2))
print("model2 n_features_in_:", model2.n_features_in_)
if hasattr(model2, 'predict_proba'):
    print("model2 has predict_proba")

print("\nInspecting rf_model.pkl:")
with open('rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)
print("Type of rf_model:", type(rf_model))
print("rf_model n_features_in_:", rf_model.n_features_in_)
print("rf_model has predict_proba:", hasattr(rf_model, 'predict_proba'))

print("\nInspecting presentloop_model3.pkl:")
with open('presentloop_model3.pkl', 'rb') as f:
    model3_data = pickle.load(f)
    model3 = model3_data['model']
print("Type of model3_data:", type(model3_data))
print("Type of model3:", type(model3))
print("model3 has predict_proba:", hasattr(model3, 'predict_proba'))

