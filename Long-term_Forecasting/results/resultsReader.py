import numpy as np

# Adjust path to your results folder
base = "/Users/kivancserefoglu/Desktop/IIT/CS584_MachineLearning/Project/Repository/S2IP-LLM/Long-term_Forecasting/results"

metrics = np.load(f"{base}/metrics.npy")
pred = np.load(f"{base}/pred.npy")
true = np.load(f"{base}/true.npy")

print("Metrics:", metrics)          # [MAE, MSE, RMSE, MAPE, MSPE]
print("Pred shape:", pred.shape)    # (samples, pred_len, number_variables)
print("True shape:", true.shape)

# To inspect a few rows:
print("Pred[0, :, 0]:", pred[0, :, 0])
print("True[0, :, 0]:", true[0, :, 0])