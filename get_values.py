import pandas as pd
import numpy as np
import re


def get_values(x):
    # Initialize index
    a = 0
    d = 0
    t = 0
    la = 0
    ha = 0
    lb = 0
    hb = 0
    lg = 0
    mg = 0

    # Initialize waveform arrays
    timestamp = []
    attention = []
    delta = []
    theta = []
    low_alpha = []
    high_alpha = []
    low_beta = []
    high_beta = []
    low_gamma = []
    mid_gamma = []

    for i in range(len(x)):
        if re.search(r"([0-9]+(:[0-9]+)+) Attention [a-zA-Z]+ [a-zA-Z]+: ", str(x.iloc[i, 0])):
            attention.append(x.iloc[i + 1, 0])
            # timestamp extraction
            timestamp.append(str(x.iloc[i, 0]).split()[0])
            a += 1

        if re.search(r"([0-9]+(:[0-9]+)+) Delta [a-zA-Z]+ [a-zA-Z]+: ", str(x.iloc[i, 0])):
            delta.append(x.iloc[i + 1, 0])
            d += 1

        if re.search(r"([0-9]+(:[0-9]+)+) Theta [a-zA-Z]+ [a-zA-Z]+: ", str(x.iloc[i, 0])):
            theta.append(x.iloc[i + 1, 0])
            t += 1

        if re.search(r"([0-9]+(:[0-9]+)+) Low Alpha [a-zA-Z]+ [a-zA-Z]+: ", str(x.iloc[i, 0])):
            low_alpha.append(x.iloc[i + 1, 0])
            la += 1

        if re.search(r"([0-9]+(:[0-9]+)+) High Alpha [a-zA-Z]+ [a-zA-Z]+: ", str(x.iloc[i, 0])):
            high_alpha.append(x.iloc[i + 1, 0])
            ha += 1

        if re.search(r"([0-9]+(:[0-9]+)+) Low Beta [a-zA-Z]+ [a-zA-Z]+: ", str(x.iloc[i, 0])):
            low_beta.append(x.iloc[i + 1, 0])
            lb += 1

        if re.search(r"([0-9]+(:[0-9]+)+) High Beta [a-zA-Z]+ [a-zA-Z]+: ", str(x.iloc[i, 0])):
            high_beta.append(x.iloc[i + 1, 0])
            hb += 1

        if re.search(r"([0-9]+(:[0-9]+)+) Low Gamma [a-zA-Z]+ [a-zA-Z]+: ", str(x.iloc[i, 0])):
            low_gamma.append(x.iloc[i + 1, 0])
            lg += 1

        if re.search(r"([0-9]+(:[0-9]+)+) Mid Gamma [a-zA-Z]+ [a-zA-Z]+: ", str(x.iloc[i, 0])):
            mid_gamma.append(x.iloc[i + 1, 0])
            mg += 1

    data_waves = ["Time", "Attention", "Delta", "Theta", "AlphaLow",
                  "AlphaHigh", "BetaLow", "BetaHigh", "GammaLow",
                  "GammaMid"]

    EEGdata = pd.DataFrame({
        "Time": timestamp,
        "Attention": attention,
        "Delta": delta,
        "Theta": theta,
        "AlphaLow": low_alpha,
        "AlphaHigh": high_alpha,
        "BetaLow": low_beta,
        "BetaHigh": high_beta,
        "GammaLow": low_gamma,
        "GammaMid": mid_gamma
    })

    EEGdata["Attention"] = pd.to_numeric(EEGdata["Attention"])
    EEGdata["Delta"] = pd.to_numeric(EEGdata["Delta"])
    EEGdata["Theta"] = pd.to_numeric(EEGdata["Theta"])
    EEGdata["AlphaLow"] = pd.to_numeric(EEGdata["AlphaLow"])
    EEGdata["AlphaHigh"] = pd.to_numeric(EEGdata["AlphaHigh"])
    EEGdata["BetaLow"] = pd.to_numeric(EEGdata["BetaLow"])
    EEGdata["BetaHigh"] = pd.to_numeric(EEGdata["BetaHigh"])
    EEGdata["GammaLow"] = pd.to_numeric(EEGdata["GammaLow"])
    EEGdata["GammaMid"] = pd.to_numeric(EEGdata["GammaMid"])

    return EEGdata