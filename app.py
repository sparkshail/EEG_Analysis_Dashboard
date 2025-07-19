import pandas as pd
from shiny import App, Inputs, Outputs, Session, reactive, render, ui
from shiny.types import FileInfo
import matplotlib.pyplot as plt
from shiny import reactive
import numpy as np
from statistics import mode
from statistics import mean
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from shiny.types import ImgData
import collections 
import matplotlib.colors as mcolors
from sklearn.ensemble import ExtraTreesClassifier
import random 

# methods from classification.py

from classification_methods import assign_result
from classification_methods import determine_result_quantile
from classification_methods import get_baselines


app_ui = ui.page_fluid(
    ui.panel_title("EEG Analysis Dashboard", "Window title"),

    ui.input_file("file1", "Choose CSV File", accept=[".csv"], multiple=False),

    ui.input_checkbox_group(
        "waves",
        "Waves",
        choices=["Delta", "Theta", "AlphaLow", "AlphaHigh", "BetaLow", "BetaHigh", "GammaLow", "GammaMid"],
        selected=["Delta", "Theta", "AlphaLow", "AlphaHigh", "BetaLow", "BetaHigh", "GammaLow", "GammaMid"],
    ),
    ui.panel_title("Data Preview", "Header title 1"),
    ui.output_table("display_data"),
    ui.panel_title("Data Statistics", "Header title 2"),
    ui.output_table("display_describe"),
    ui.input_checkbox_group(
        "stats",
        "Summary Stats",
        choices=["Row Count", "Column Count", "Column Names"],
        selected=["Row Count", "Column Count", "Column Names"],
    ),
    ui.output_table("summary"),
    ui.output_table("event_info"),

    ui.input_select(
        "select",
        "Choose an analysis you would like to run:",
        {
            "Waveform Analysis": {"TH": "Thresholds", "FI": "Feature Importance"},
            "Classification": {"ML": "Multiclassification"},
        },
    ),
    ui.output_plot("analysis")
)


def server(input: Inputs, output: Outputs, session: Session):
    @reactive.calc
    def parsed_file():
        file: list[FileInfo] | None = input.file1()
        if file is None:
            return pd.DataFrame()
        return pd.read_csv(  # pyright: ignore[reportUnknownMemberType]
            file[0]["datapath"]
        )
    
    @reactive.calc
    def waves():
        df = parsed_file()
        if df.empty:
            return pd.DataFrame()
        keep = input.waves()
        for col in df.columns:
            if col not in keep:
                df = df.drop(col, axis=1)
        return df
        #return df.loc[:, input.waves()]


    @render.table
    def display_data():
        df = waves()
        return df.head()


    @render.table
    def display_describe():
        #df = parsed_file()
        df = waves()
        stats = ["Count", "Mean", "Standard Deviation", "Min",  "25%", "75%", "50%", "Max"]
        summary = df.describe()
        summary.insert(0,"Stat", stats)
        return summary
    
    @render.plot
    def plot_waves(df):
        #df = parsed_file()
        df.plot(title="Wave Measures Over Time")
        plt.legend(bbox_to_anchor=(1, 1))


    @render.data_frame
    def ML(df):
        df =  df.drop("Event",axis=1)
        classified = assign_result(df)
        return classified.head()
    
    @reactive.calc
    def multi_class():
        df = waves()
        classified = assign_result(df)[0]
        all_events = classified["Event"]
        limit = len(classified.columns)-1
        #limit = 8
        y = classified.iloc[:, limit]
        X = classified.iloc[:, :limit]
        return[classified, all_events, X, y, df]


    @render.table
    def event_info():
        data = multi_class()
        events = data[1]
        string_events = []
        event_info = {"Number of Traumatic Periods": [0], "Average Traumatic Period Length (sec)": [0],
                      "Total Time for Non-Trauma Experiences": [0], "Total Time for Trauma Experiences":[0], 
                       "% Non-Trauma": ["0%"], "% Low": ["0%"], "% Medium": ["0%"], "% High": ["0%"]}
        for i in range(0,len(events)):
            if events[i] == "000":
                string_events.append("Non-Trauma")
            elif events[i] == "100":
                string_events.append("Low")
            elif events[i] == "010":
                string_events.append("Medium")
            else:
                string_events.append("High")

        timer = 0
        event_lengths = []
        
        for i in range(0, len(string_events)-1):
            if string_events[i] == "Low" or string_events[i] == "Medium" or string_events[i] == "High":
                timer = timer + 1
            if string_events[i+1] == "Non-Trauma" and string_events[i]!= "Non-Trauma":
                event_lengths.append(timer)
                timer = 0

        c = collections.Counter(string_events)
        y = [value for key,value in c.items()]
        total = sum(value for key,value in c.items())
        non_trauma_total = c["Non-Trauma"]
        trauma_total = total - non_trauma_total

        percentages = []

        for value in y:
            per = round((value/total),3)*100
            percentages.append(per)

        try:
            event_info["% Non-Trauma"] = [percentages[0]]
        except:
            event_info["% Non-Trauma"] = ["0%"]
        try:
            event_info["% Low"] = [percentages[1]]
        except:
            event_info["% Low"] = ["0%"]
        try:
            event_info["% Medium"] = [percentages[2]]
        except:
            event_info["% Medium"] = ["0%"]
        try:
            event_info["% High"] = [percentages[3]]
        except:
            event_info["% High"] = ["0%"]


        num_trauma = len(event_lengths)
        event_info["Number of Traumatic Periods"] = [num_trauma]
        event_info["Average Traumatic Period Length (sec)"] = [mean(event_lengths)]
        event_info["Total Time for Non-Trauma Experiences"] = [non_trauma_total]
        event_info["Total Time for Trauma Experiences"] = [trauma_total]

        stats = pd.DataFrame(event_info)

        return stats.head() 


    @render.plot
    def analysis():
        
        #df = parsed_file()
        #df = waves()
        #test = df 
        #selection = input.select()
    

        #df =  df.drop("Event",axis=1)
        #classified = assign_result(test)[0]
        #all_events = classified["Event"]
        #limit = len(classified.columns)-1
        #limit = 8
        #y = classified.iloc[:, limit]
       # X = classified.iloc[:, :limit]


        # to be used for stats 
        num_trauma  = 0
        num_non_trauma = 0
        trauma = False  


       #-------------------------------
        data = multi_class()
        classified = data[0]
        all_events = data[1]
        X = data[2]
        y = data[3]
        df = data[4]

       #--------------------------------

        selection = input.select()
        
        if selection == "TH":

            baselines = get_baselines(X)
            for i in range(0,len(df.columns)-1): # current wave
                reg_x = []
                reg_y = []
                bold_x = []
                bold_y = []
                wave = df.columns[i]
                for j in range(0,len(df)):  # current instance
                    if df[wave][j] > baselines[i]: 
                        bold_x.append(j)
                        bold_y.append(df[wave][j])
                        reg_x.append(j)
                        reg_y.append(df[wave][j])
                    else:
                        reg_x.append(j)
                        reg_y.append(df[wave][j])
                plt.plot(reg_x, reg_y, label = df.columns[i])
                plt.plot(bold_x, bold_y, 'ko', color = "black", markersize=.75)
            plt.legend(prop={'size': 6})
            plt.ylabel("Intensity")
            plt.xlabel("Time")
        elif selection == "FI":
            model = ExtraTreesClassifier(random_state=2)
            model.fit(X,y)
            feat_importances = pd.Series(model.feature_importances_, index=X.columns)
            feat_importances.nlargest(5).plot(kind='barh')
            plt.title("Feature Importance")
        elif selection == "ML":
            bar_colors = ["g","y","orange","red"]
            string_events = []
            for i in range(0,len(all_events)):
                if all_events[i] == "000":
                    #all_events[i] = "Non-Trauma"
                    string_events.append("Non-Trauma")
                    #bar_colors.extend("g")
                elif all_events[i] == "100":
                    #all_events[i] = "Low"
                    string_events.append("Low")
                    #bar_colors.extend("y")
                elif all_events[i] == "010":
                    #all_events[i] = "Medium"
                    string_events.append("Medium")
                   # bar_colors.extend("r")
                else:
                    #all_events[i] = "High"
                    string_events.append("High")
                   # bar_colors.extend("r")

        
            c = collections.Counter(string_events)

            x = [key for key,value in c.items()]
            y = [value for key,value in c.items()]
            bar_colors =["g","y","orange","red"]
            total = sum(value for key,value in c.items())
            percentages = []

            for value in y:
                per = round((value/total),3)*100
                percentages.append(per)

            pps = plt.bar(x,y, color = bar_colors)
            num = 0
            for p in pps:
                plt.text(x=p.get_x() + p.get_width() / 2, y=p.get_height()+.20,
                    s="{}%".format(percentages[num]), ha='center')
                num = num+1

            plt.title("Event Intensity")
            plt.ylabel("Number of Instances")

        return 
    
    @render.text
    def value():
        return "You choose: " + str(input.select())


    @render.table
    def summary():
        df = waves()
        
        if df.empty:
            return pd.DataFrame()
        
        #display_data(df)

        # Get the row count, column count, and column names of the DataFrame
        row_count = df.shape[0]
        column_count = df.shape[1]
        names = df.columns.tolist()
        if "Event" in names: 
            df = df.drop("Event", axis =1)
        column_names = ", ".join(str(name) for name in names)

        # Create a new DataFrame to display the information
        info_df = pd.DataFrame(
            {
                "Row Count": [row_count],
                "Column Count": [column_count],
                "Column Names": [column_names]
            }
        )

        # input.stats() is a list of strings; subset the columns based on the selected
        # checkboxes
        #return df.head()
        return info_df.loc[:, input.stats()]
       # return df.describe()


app = App(app_ui, server)

