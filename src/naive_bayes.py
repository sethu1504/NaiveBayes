import pandas as pd
import numpy as np
import math


data = pd.read_csv('../datasets/hockey.csv')  # Load data

# Drop unnecessary columns
del data['id']
del data['PlayerName']
del data['Country']
del data['rs_PlusMinus']
del data['sum_7yr_GP']
del data['sum_7yr_TOI']

# SPLIT DATA
train_data = data[data['DraftYear'] <= 2000]
test_data = data[data['DraftYear'] == 2001]

# Calculate Prior probabilities
tot_rows = float(train_data.shape[0])
yes_count = train_data[train_data['GP_greater_than_0'] == 'yes'].shape[0]
no_count = train_data[train_data['GP_greater_than_0'] == 'no'].shape[0]
prior_yes_prob = yes_count / tot_rows
prior_no_prob = no_count / tot_rows


def calculate_prob(mean, std, x):
    exp = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(std, 2))))
    return (1 / (math.sqrt(2 * math.pi) * std)) * exp


def calculate_discrete_variable_prob(attr, val):
    y_count = train_data[(train_data['GP_greater_than_0'] == 'yes') & (train_data[attr] == val)].shape[0]
    n_count = train_data[(train_data['GP_greater_than_0'] == 'no') & (train_data[attr] == val)].shape[0]
    return y_count / tot_rows, n_count / tot_rows


# Calculate mean and std for each attribute w.r.t class label
continuous_attr_yes_dict = {}
continuous_attr_no_dict = {}
continuous_attributes = list(train_data.dtypes[train_data.dtypes != 'object'].index)
for attribute in continuous_attributes:
    np_arr_yes = np.array(train_data[train_data['GP_greater_than_0'] == 'yes'][attribute])
    np_arr_no = np.array(train_data[train_data['GP_greater_than_0'] == 'no'][attribute])
    continuous_attr_yes_dict[attribute] = (np.mean(np_arr_yes), np.std(np_arr_yes))
    continuous_attr_no_dict[attribute] = (np.mean(np_arr_no), np.std(np_arr_no))

# Take class labels separately and remove it from test data
class_labels = test_data['GP_greater_than_0']
del test_data['GP_greater_than_0']

# MAKE PREDICTIONS
correct_predictions = 0
wrong_predictions = 0
test_records = test_data.shape[0]
for i in range(test_records):
    row = test_data.iloc[i]
    yes_prob = 1
    no_prob = 1
    for attribute, value in row.iteritems():
        if attribute in continuous_attr_yes_dict:
            yes_mean = continuous_attr_yes_dict.get(attribute)[0]
            yes_std = continuous_attr_yes_dict.get(attribute)[1]
            yes_prob *= calculate_prob(yes_mean, yes_std, value)
            no_mean = continuous_attr_no_dict.get(attribute)[0]
            no_std = continuous_attr_no_dict.get(attribute)[1]
            no_prob *= calculate_prob(no_mean, no_std, value)
        else:
            discrete_probabilities = calculate_discrete_variable_prob(attribute, value)
            yes_prob *= discrete_probabilities[0]
            no_prob *= discrete_probabilities[1]
    # Multiply by prior probability
    yes_prob *= prior_yes_prob
    no_prob *= prior_no_prob

    predicted_class = 1 if yes_prob > no_prob else 0
    actual_class = class_labels.values[i]
    if (predicted_class == 0 and actual_class == 'no') or (predicted_class == 1 and actual_class == 'yes'):
        correct_predictions += 1
    else:
        wrong_predictions += 1

print 'Correct Predictions = ' + str(correct_predictions)
print 'Wrong Predictions = ' + str(wrong_predictions)
print 'Accuracy = ' + str(correct_predictions / float(test_records))
