private_testing_alert_key = np.array(private_testing_alert_key)
testing_alert_key = np.concatenate((testing_alert_key, private_testing_alert_key), axis = 0)

import xgboost as xgb
xgbrModel=xgb.XGBClassifier(learning_rate=0.02, max_depth=6, scale_pos_weight=20, objective="reg:squarederror")
xgbrModel.fit(train_set, train_labels)

# Read csv of all alert keys need to be predicted
data_dir = './ITF_data/'

public_private_test_csv = os.path.join(data_dir, '預測的案件名單及提交檔案範例.csv')
df_public_private_test = pd.read_csv(public_private_test_csv)

# Predict probability
predicted = []
for i, _x in enumerate(xgbrModel.predict_proba(testing_data)):
    predicted.append([testing_alert_key[i], (_x[1]+1) / 2])
predicted = sorted(predicted, reverse=True, key= lambda s: s[1])

# 考慮private alert key部分，滿足上傳條件

public_private_alert_key = df_public_private_test['alert_key'].values
print(len(public_private_alert_key))

# For alert key not in public, add zeros

for key in public_private_alert_key:
    if key not in testing_alert_key:
        predicted.append([key, 0])

predict_alert_key, predict_probability = [], []
for key, prob in predicted:
    predict_alert_key.append(key)
    predict_probability.append(prob)

df_predicted = pd.DataFrame({
    "alert_key": predict_alert_key,
    "probability": predict_probability
})

df_predicted.to_csv('./ITF_data/results/prediction_xgb_test_172.csv', index=False)