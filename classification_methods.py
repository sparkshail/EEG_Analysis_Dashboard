

# load necessary libraries

import pandas as pd
import numpy as np
from statistics import mode
from statistics import mean
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


# splits data into train and test sets
# calc accuracy of NeuralNet
def get_accuracy(df):
    y = df.iloc[:, len(df.columns) - 1]
    X = df.iloc[:, :(len(df.columns) - 1)]
    # was using random state 0
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # was using random state 115
    NN = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=115)
    NN.fit(X_train, y_train)

    accuracy = round(NN.score(X_test, y_test), 4)
    return accuracy


# split data 80/20
def split_data(data):
    #y = data.iloc[:, 8]
    #X = data.iloc[:, :8]
    y = data.iloc[:, len(data.columns) - 1]
    X = data.iloc[:, :(len(data.columns) - 1)]
    # Split training and testing 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2)
    return [X_train, X_test, y_train, y_test]


# determine baseline values
def get_baselines(cur_data):
    baseline_values = []
    epoch_means = cur_data.mean()
    epoch_stds = cur_data.std()
    #for i in range(0, 8):
    for i in range(0, len(cur_data.columns)):
        baseline_values.append(epoch_means[i] + (1 * epoch_stds[i]))

    return baseline_values


# add 'event' feature to observation
# the cur_data is one epoch
# assigns results from 'determine_result_quantile'
def assign_result(cur_data):
    #baselines = get_baselines(cur_data)

    epoch_means = cur_data.mean()
    epoch_stds = cur_data.std()
    cur_data['Event'] = ""
    thresholds = []
    for i in range(0, len(epoch_means)):
        thresholds.append(epoch_means[i] + (1 * epoch_stds[i]))

    for i in range(0, len(cur_data)):
        cur_row = cur_data.iloc[i]
        # cur_result = determine_result(cur_row,epoch_means,epoch_stds)
        cur_result = determine_result_quantile_shiny(cur_data.columns, cur_row, epoch_means, epoch_stds)  # quantile result
        #cur_data.iloc[[i], [8]] = cur_result
        cur_data.iloc[[i], [len(cur_data.columns)-1]] = cur_result
    # print(cur_data)
    return [cur_data, thresholds, epoch_means, epoch_stds]


# old method that used binary classification
# determine if an observation is traumatic or non-traumatic
def determine_result(row, means, stds):
    wave_decisions = {'Delta': False, 'Theta': False, 'AlphaLow': False, 'AlphaHigh': False,
                      'BetaLow': False, 'BetaHigh': False, 'GammaLow': False, 'GammaMid': False}

    count = 0
    for key, value in wave_decisions.items():
        wave = key
        # cur_wave_mean = calc_wave_row_mean(row,wave)
        cur_wave = row[wave]
        if cur_wave > (means[count] + (1 * stds[count])):
            wave_decisions[wave] = True
        count = count + 1

    num_true = sum(value == True for value in wave_decisions.values())
    if (num_true >= 4):
        result = 'Trauma'
    else:
        result = 'Non-Trauma'

    return result


# updated method that uses multiclassification
# non-trauma, low, medium, or high
def determine_result_quantile(row, means, stds):
    wave_decisions = {'Delta': "", 'Theta': "", 'AlphaLow': "", 'AlphaHigh': "",
                      'BetaLow': "", 'BetaHigh': "", 'GammaLow': "", 'GammaMid': ""}

    count = 0
    for key, value in wave_decisions.items():
        wave = key
        cur_wave = row[wave]
        cur_mean = means.iloc[count]

        if cur_wave >= (cur_mean * .75) + cur_mean:
            wave_decisions[wave] = 'High'
            # wave_decisions[wave] = .75
           # print("high")
        elif cur_wave > (cur_mean * .5) + cur_mean:
            wave_decisions[wave] = 'Medium'
            # wave_decisions[wave] = .5
            #print("medium")
        elif cur_wave > (cur_mean * .25) + cur_mean:
            wave_decisions[wave] = 'Low'
            # wave_decisions[wave] = .25
            #print("low")
        else:
            wave_decisions[wave] = 'Non-Trauma'
        count = count + 1

    result = mode(wave_decisions.values())
    #print(result)
   
    num_high = sum(value == 'High' for value in wave_decisions.values())
    if (num_high >= 4):
        # result = 'High'
        result = "001"
    elif (sum(value == 'Medium' for value in wave_decisions.values()) + sum(
            value == 'High' for value in wave_decisions.values()) >= 4):
        # result = 'Medium'
        result = "010"
    elif (sum(value == 'Low' for value in wave_decisions.values()) + sum(
            value == 'Medium' for value in wave_decisions.values()) >= 4):
        # result = 'Low'
        result = "100"
    else:
        # result = "Non-Trauma"
        result = "000"

    return result




#################################################################################################
# updated method that uses multiclassification
# non-trauma, low, medium, or high
# UPDATED FOR SHINY APP
#################################################################################################
def determine_result_quantile_shiny(waves, row, means, stds):
    #wave_decisions = {'Delta': "", 'Theta': "", 'AlphaLow': "", 'AlphaHigh': "",
                     # 'BetaLow': "", 'BetaHigh': "", 'GammaLow': "", 'GammaMid': ""}
    wave_decisions = {}

    for i in range(0,len(waves)-1):
        wave_decisions[waves[i]] = ""

    count = 0
    for key, value in wave_decisions.items():
        wave = key
        cur_wave = row[wave]
        cur_mean = means.iloc[count]

        if cur_wave >= (cur_mean * .75) + cur_mean:
            wave_decisions[wave] = 'High'
            # wave_decisions[wave] = .75
           # print("high")
        elif cur_wave > (cur_mean * .5) + cur_mean:
            wave_decisions[wave] = 'Medium'
            # wave_decisions[wave] = .5
            #print("medium")
        elif cur_wave > (cur_mean * .25) + cur_mean:
            wave_decisions[wave] = 'Low'
            # wave_decisions[wave] = .25
            #print("low")
        else:
            wave_decisions[wave] = 'Non-Trauma'
        count = count + 1

    result = mode(wave_decisions.values())
    #print(result)
   
    num_high = sum(value == 'High' for value in wave_decisions.values())
    limit = len(waves)/2
    if (num_high >= limit):
        # result = 'High'
        result = "001"
    elif (sum(value == 'Medium' for value in wave_decisions.values()) + sum(
            value == 'High' for value in wave_decisions.values()) >= limit):
        # result = 'Medium'
        result = "010"
    elif (sum(value == 'Low' for value in wave_decisions.values()) + sum(
            value == 'Medium' for value in wave_decisions.values()) >= limit):
        # result = 'Low'
        result = "100"
    else:
        # result = "Non-Trauma"
        result = "000"

    return result


# writing an updated split_epochs
def split_epoch(model, data,max_n):
    for i in range(1, max_n):
        data_chunks = np.array_split(data, i)
        for j in range(0, len(data_chunks)):
            classified = determine_result_quantile(data_chunks[j])
            train_test = split_data(classified)
            model.partial_fit(train_test[0],train_test[1])

    return model



# max_e is max amount of epochs to iterate
# iterate n from 1..max_e
# at each iteration, data is split into n sections
# a model is then built on each n_i section, and its accuracy is calculated (1 epoch)
# results from all iterations are averaged
def split_epochs(data, max_e):
    df = pd.DataFrame(columns=['SplitNum', 'EpochNum', 'Thresholds', 'PercentBelow', 'PercentAbove', 'Accuracy'])

    mean_accuracies = []
    size = len(data)
    for i in range(1, max_e):
        accuracies = []
        # mean_accuracies = []
        data_chunks = np.array_split(data, i)
        # start_index,end_index = 0,0
        for j in range(0, len(data_chunks)):
            classified = assign_result(data_chunks[j])
            class_data = classified[0]
            thresh = classified[1]
            means = classified[2]
            stds = classified[3]

            event_counts = class_data['Event'].value_counts()
            per_below = (event_counts[0]) / len(class_data)
            per_above = 1 - per_below
            cur_acc = get_accuracy(class_data)

            df = pd.concat(
                [df, pd.DataFrame([{'SplitNum': i, 'EpochNum': j, 'Thresholds': thresh, 'Means': means, 'SD': stds,
                                    'PercentBelow': per_below, 'PercentAbove': per_above,
                                    'Accuracy': cur_acc}])], ignore_index=True)
            accuracies.append(cur_acc)
        mean_accuracies.append(mean(accuracies))
        print("Number of epochs: ", i, "Accuracy: ", mean(accuracies))

    return [df, mean_accuracies]
