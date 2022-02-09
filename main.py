"""
Code part of final project for UCD Specialist Certificate in Data Analytics
Prepared by Peter Morgan
Course lecturer Shruti Bansal
Date submitted 09 Feb 2022

This code file is submitted in conjunction with a final Report.

Encapsulating the entire process of analysis into a single executable file can obscure the interactive
nature of the process. It is challenging to demonstrate the constant inspection of various objects such as dataframes
as part of an ongoing process of refinement. For this reason I have included functions to handle inspection of
data and called after the processing has been completed should the user want to do this.
These function calls can be commented out or not, depending on the users requirements. All outputs from these are
included in the Appendix of the Report.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.linear_model import LinearRegression
from sklearn import metrics


def print_dataframe_details(Incoming_Dataframe_Name, Incoming_Dataframe):
    print("\nDetails for %s" %(Incoming_Dataframe_Name))
    print('\nShape:\n')
    print(Incoming_Dataframe.shape)
    print('\nDataframe Info:\n')
    print(Incoming_Dataframe.info())
    print('\nDataframe head:\n')
    print(Incoming_Dataframe.head())
    return 0

def write_dataframe_to_disk(File_Name, Incoming_Dataframe):
    # PyCharm output tends to abbreviate dataframes. This function was used to get
    # dataframes into a format easily included into the report
    Incoming_Dataframe.to_csv(r"C:\Users\Peter\Desktop\UCD\project\world happiness report\%s" %(File_Name))
    return 0

# Names of files on disk - alter as required if file location or name is changed
File_on_disk_20x = r"C:\Users\Peter\Desktop\UCD\project\world happiness report\DataPanelWHR2021C2.csv"
File_on_disk__2021 = r"C:\Users\Peter\Desktop\UCD\project\world happiness report\DataForFigure2.1WHR2021C2.csv"
File_on_disk_mortality_data = r"C:\Users\Peter\Desktop\UCD\project\world happiness report\MortalityDataWHR2021C2.csv"

# Create Raw dataframes
Raw_WHR_20x = pd.read_csv(File_on_disk_20x)
Raw_WHR_2021 = pd.read_csv(File_on_disk__2021)
Mortality_data = pd.read_csv(File_on_disk_mortality_data)
Median_age_data = Mortality_data[['Country name', 'Median age']]


# Create Clean DataFrames - see report for details and justifications

# Clean Raw_WHR_20x
Clean_WHR_20x = Raw_WHR_20x.drop(columns=['Positive affect', 'Negative affect'])
Clean_WHR_20x.dropna(inplace=True)

# Clean Raw_WHR_2021
Clean_WHR_2021 = Raw_WHR_2021.drop(columns=
                                   ['Regional indicator',
                                    'Standard error of ladder score',
                                    'upperwhisker',
                                    'lowerwhisker',
                                    'Ladder score in Dystopia',
                                    'Explained by: Log GDP per capita',
                                    'Explained by: Social support',
                                    'Explained by: Healthy life expectancy',
                                    'Explained by: Freedom to make life choices',
                                    'Explained by: Generosity',
                                    'Explained by: Perceptions of corruption',
                                    'Dystopia + residual'])

Clean_WHR_2021.insert(1, 'Year', 2021)


# Match columns and Combine DataFrames, sort and reset index
Clean_WHR_20x.columns = Clean_WHR_2021.columns
Combined_WHR = pd.concat([Clean_WHR_2021, Clean_WHR_20x])
Combined_WHR.sort_values(by=['Country name', 'Year'], inplace=True)
Combined_WHR.reset_index(inplace=True, drop=True)

# Create and Clean Age_and_happiness dataframe
Age_and_happiness = pd.merge(Clean_WHR_2021[['Country name', 'Ladder score']], Median_age_data, on='Country name')
Age_and_happiness.dropna(inplace=True)


# Split in to features (x) and target (y). Columns can be commented out of x to investigate effects
# of multicollinearity
x_columns = ['Logged GDP per capita',
             'Social support',
             'Healthy life expectancy',
             'Freedom to make life choices',
             'Generosity',
             'Perceptions of corruption']
x = Combined_WHR[x_columns]
y = Combined_WHR['Ladder score']


# split into training and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, shuffle=True, random_state=32)



# Create Regression object
WHR_Model = ensemble.GradientBoostingRegressor(
    n_estimators=100, #100
    learning_rate=0.1,
    max_depth=5,#30,
    min_samples_split=4,
    min_samples_leaf=6,
    max_features=0.6,
    loss='huber')


# train model
WHR_Model.fit(x_train, y_train)
# test model
y_pred = WHR_Model.predict(x_test)

training_data_mae = metrics.mean_absolute_error(y_train, WHR_Model.predict(x_train))
test_data_mae = metrics.mean_absolute_error(y_test, WHR_Model.predict(x_test))
training_data_r2 = metrics.r2_score(y_train, WHR_Model.predict(x_train))
test_data_r2 = metrics.r2_score(y_test, WHR_Model.predict(x_test))

# print section


Decision = input('Would you like to print view details from various stages of process? Y = yes, any other key to exit')
if (Decision == 'y' or Decision == 'Y'):
    print("\nTraining data Mean Absolute Error: %.5f" % training_data_mae)
    print("Test data Mean Absolute Error: %.5f" % test_data_mae)
    print("\nTraining data r2: %.5f" % training_data_r2)
    print("Test data r2: %.5f" % test_data_r2)

    # Comment / Uncomment as required, otherwise see Appendix Fig 1 to Fig 3 in report
    print_dataframe_details('Raw_WHR_20x', Raw_WHR_20x)
    print_dataframe_details('Raw_WHR_2021', Raw_WHR_2021)
    print_dataframe_details('Combined_WHR', Combined_WHR)

    # Uncomment should you wish to write dataframes to disk
    # write_dataframe_to_disk('Raw_WHR20x.csv', Raw_WHR_20x.head())
    # write_dataframe_to_disk('Raw_WHR_2021.csv', Raw_WHR_2021.head())
    # write_dataframe_to_disk('Combined_WHR.csv', Combined_WHR.head())

    # Check for multicollinearity

    pd.plotting.scatter_matrix(x, figsize=([8, 8]));
    plt.show()
    Correlation_Matrix = x.corr()
    Correlation_Matrix.head()
    # write_dataframe_to_disk('Correlation_Matrix.csv', Correlation_Matrix)

    # Plot Ladder score against median age and calculate correlation between the two
    x_coord = np.array(Age_and_happiness['Median age'])
    y_coord = np.array(Age_and_happiness['Ladder score'])

    plt.scatter(x_coord, y_coord)
    plt.title("Ladder score and Median age")
    plt.xlabel("Median age")
    plt.ylabel("Ladder score")
    plt.show()
    Ladder_age_corr_coef = np.corrcoef(x_coord, y_coord)

    print('Correlation coefficient for Ladder and Median age: ', Ladder_age_corr_coef[0, 1])

else:
    print('Thank you and goodbye')