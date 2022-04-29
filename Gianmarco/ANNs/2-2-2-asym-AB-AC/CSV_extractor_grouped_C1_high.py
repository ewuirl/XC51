import sys
import numpy as np

sys.path.append("../../../../")
from simCRN.multivariate_reg import (
    read_eq_data_file,
)
from sklearn.model_selection import train_test_split
import pandas as pd

FILENAME = "2-2-2-asym-AB-AC"  # Note: Adjust this as needed.
Ci_all_array, Am_array, Cmin, Cmax, Ai = read_eq_data_file(f"../../../{FILENAME}.txt")

# I will use this script to convert things into CSV format.
# This is because my TensorFlow workflows take CSVs.

# 70% train, 10% validation, and 20% test
cutoff = np.percentile(
    Ci_all_array[:, 0], 80
)  # Note: Change that index depending if you want C1 or C2. Change the percentile value depending if you want high or low.
print(f"The cutoff is {cutoff}")
indices = (
    Ci_all_array[:, 0] > cutoff
)  # Note: Change that index depending if you want C1 or C2. Change the inequality depending if you want high or low.
shadow_indices = np.logical_not(indices)

X_test = Am_array[indices, :]
y_test = Ci_all_array[indices, :]
X_train = Am_array[shadow_indices, :]
y_train = Ci_all_array[shadow_indices, :]

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.125, random_state=0
)

num_train = X_train.shape[0]
num_val = X_val.shape[0]
num_test = X_test.shape[0]

# Putting together a Pandas dataframe and saving it as a CSV file.
data = {
    "A1": np.concatenate((X_train[:, 0], X_val[:, 0], X_test[:, 0])),
    "A2": np.concatenate((X_train[:, 1], X_val[:, 1], X_test[:, 1])),
    "C1": np.concatenate((y_train[:, 0], y_val[:, 0], y_test[:, 0])),
    "C2": np.concatenate((y_train[:, 1], y_val[:, 1], y_test[:, 1])),
    "bin": ["train"] * num_train + ["val"] * num_val + ["test"] * num_test,
}
my_df = pd.DataFrame(data=data)

my_df.to_csv(
    f"{FILENAME}_grouped_C1_high.csv", index=False
)  # Note: Change file name as needed.
