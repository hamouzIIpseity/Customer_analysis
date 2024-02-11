
import os
import numpy as np 
import pandas as pd 
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

df = pd.read_csv("C:/Hamza/project kaggle/Customer_personality_analysis/data/marketing_campaign.csv")
df.drop(['Z_CostContact', 'Z_Revenue', 'ID'], axis=1, inplace=True)
df[df['Income'].isnull()].head()
df.dropna(inplace=True)
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format = "%d-%m-%Y")
df['Age'] = 2015 - df['Year_Birth']
df.drop(['Year_Birth'], axis=1, inplace=True)

def classify_numerical_features(df, threshold=10, excluded_feature=None):
    continuous_features = []
    discrete_features = []

    for column in df.columns: # select numerical features
        # Skip the excluded_feature
        if column == excluded_feature:
            continue
        
        unique_values = df[column].nunique() # count number of unique values in each columns
        if unique_values < threshold or ('Num' in column):
            discrete_features.append(column) 
        else: # if more thant 10 differents values, it is considered as a continuous feature
            continuous_features.append(column)

    return continuous_features, discrete_features

continuous, discrete = classify_numerical_features(df, excluded_feature='Dt_Customer')




# Define the number of plots per row
plots_per_row = 2

# Calculate the number of rows required
num_rows = len(continuous) // plots_per_row
if len(continuous) % plots_per_row != 0:
    num_rows += 1

# Create subplots
fig, axes = plt.subplots(num_rows, plots_per_row, figsize=(12, 4 * num_rows))

# Flatten the axes array to simplify indexing
axes = axes.flatten()

# Plot each continuous feature
for i, column in enumerate(continuous):
    ax = axes[i]
    
    # Plot KDE plot
    sns.histplot(df[column], kde=True, color='purple', bins=20, fill=True, ax=ax)
    
    # Add counts values to each bar
    for p, value in zip(ax.patches, df[column].value_counts().sort_index().index):
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                    ha='center', va='center', fontsize=8, color='white')

    # Set labels and title
    ax.set_xlabel(column)
    ax.set_ylabel('Density')
    ax.set_title(f'{column} distribution')

# Adjust layout to prevent overlapping titles
plt.tight_layout()

# Show the plots
plt.show()



# Boxplots segmented by Marital_Status
warnings.filterwarnings("ignore", category=FutureWarning)

discrete_column = 'Marital_Status'  # Replace with the actual column name

# Define the number of plots per row
plots_per_row = 2

# Calculate the number of rows required
num_rows = len(continuous) // plots_per_row
if len(continuous) % plots_per_row != 0:
    num_rows += 1

# Create subplots
fig, axes = plt.subplots(num_rows, plots_per_row, figsize=(12, 4 * num_rows))

# Flatten the axes array to simplify indexing
axes = axes.flatten()

# Plot each continuous feature segmented by 'Marital_Status'
for i, column in enumerate(continuous):
    ax = axes[i]
    
    # Plot boxplot
    sns.boxplot(x=discrete_column, y=column, data=df, ax=ax, palette='plasma')
    
    # Set labels and title
    ax.set_xlabel(discrete_column)
    ax.set_ylabel(column)
    ax.set_title(f'{column} distribution by {discrete_column}')

# Adjust layout to prevent overlapping titles
plt.tight_layout()

# Show the plots
plt.show()



# Boxplots segmented by Education
warnings.filterwarnings("ignore", category=FutureWarning)

discrete_column = 'Education'  # Replace with the actual column name

# Define the number of row
plots_per_row = 2

# Calculate the number of rows required
num_rows = len(continuous) // plots_per_row
if len(continuous) % plots_per_row != 0:
    num_rows += 1

# Create subplots
fig, axes = plt.subplots(num_rows, plots_per_row, figsize=(12, 4 * num_rows))

# Flatten the axes array to simplify indexing
axes = axes.flatten()

# Plot each continuous feature
for i, column in enumerate(continuous):
    ax = axes[i]
    
    # Plot boxplot
    sns.boxplot(x=discrete_column, y=column, data=df, ax=ax, palette='plasma')
    
    # Set labels and title
    ax.set_xlabel(discrete_column)
    ax.set_ylabel(column)
    ax.set_title(f'{column} distribution by {discrete_column}')

# Adjust layout to prevent overlapping titles
plt.tight_layout()

# Show the plots
plt.show()




# Boxplots segmented by Kidhome
warnings.filterwarnings("ignore", category=FutureWarning)

discrete_column = 'Kidhome'  # Replace with the actual column name

# Define the number of row
plots_per_row = 2

# Calculate the number of rows required
num_rows = len(continuous) // plots_per_row
if len(continuous) % plots_per_row != 0:
    num_rows += 1

# Create subplots
fig, axes = plt.subplots(num_rows, plots_per_row, figsize=(12, 4 * num_rows))

# Flatten the axes array to simplify indexing
axes = axes.flatten()

# Plot each continuous feature
for i, column in enumerate(continuous):
    ax = axes[i]
    
    # Plot boxplot
    sns.boxplot(x=discrete_column, y=column, data=df, ax=ax, palette='plasma')
    
    # Set labels and title
    ax.set_xlabel(discrete_column)
    ax.set_ylabel(column)
    ax.set_title(f'{column} distribution by {discrete_column}')

# Adjust layout to prevent overlapping titles
plt.tight_layout()

# Show the plots
plt.show()





# Boxplots segmented by Teenhome
warnings.filterwarnings("ignore", category=FutureWarning)

discrete_column = 'Teenhome'  # Replace with the actual column name

# Define the number of row
plots_per_row = 2

# Calculate the number of rows required
num_rows = len(continuous) // plots_per_row
if len(continuous) % plots_per_row != 0:
    num_rows += 1

# Create subplots
fig, axes = plt.subplots(num_rows, plots_per_row, figsize=(12, 4 * num_rows))

# Flatten the axes array to simplify indexing
axes = axes.flatten()

# Plot each continuous feature
for i, column in enumerate(continuous):
    ax = axes[i]
    
    # Plot boxplot
    sns.boxplot(x=discrete_column, y=column, data=df, ax=ax, palette='plasma')
    
    # Set labels and title
    ax.set_xlabel(discrete_column)
    ax.set_ylabel(column)
    ax.set_title(f'{column} distribution by {discrete_column}')

# Adjust layout to prevent overlapping titles
plt.tight_layout()

# Show the plots
plt.show()




# Boxplots segmented by Complain
warnings.filterwarnings("ignore", category=FutureWarning)

discrete_column = 'Complain'  # Replace with the actual column name

# Define the number of row
plots_per_row = 2

# Calculate the number of rows required
num_rows = len(continuous) // plots_per_row
if len(continuous) % plots_per_row != 0:
    num_rows += 1

# Create subplots
fig, axes = plt.subplots(num_rows, plots_per_row, figsize=(12, 4 * num_rows))

# Flatten the axes array to simplify indexing
axes = axes.flatten()

# Plot each continuous feature
for i, column in enumerate(continuous):
    ax = axes[i]
    
    # Plot boxplot
    sns.boxplot(x=discrete_column, y=column, data=df, ax=ax, palette='plasma')
    
    # Set labels and title
    ax.set_xlabel(discrete_column)
    ax.set_ylabel(column)
    ax.set_title(f'{column} distribution by {discrete_column}')

# Adjust layout to prevent overlapping titles
plt.tight_layout()

# Show the plots
plt.show()




import seaborn as sns
import matplotlib.pyplot as plt

# Define the number of plots per row
plots_per_row = 2

# Calculate the number of rows required
num_rows = len(discrete) // plots_per_row
if len(discrete) % plots_per_row != 0:
    num_rows += 1

# Create subplots
fig, axes = plt.subplots(num_rows, plots_per_row, figsize=(12, 4 * num_rows))

# Flatten the axes array to simplify indexing
axes = axes.flatten()

# Plot each discrete feature with countplot using the default color palette
for i, column in enumerate(discrete):
    ax = axes[i]
    
    # Plot countplot with default color palette
    sns.countplot(x=column, data=df, ax=ax, order=df[column].value_counts().index, color='purple')
    
    # Add counts values to each bar
    for p, value in zip(ax.patches, df[column].value_counts().sort_index().index):
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                    ha='center', va='center', fontsize=8, color='white')

    # Set labels and title
    ax.set_xlabel(column)
    ax.set_ylabel('Count')
    ax.set_title(f'{column} distribution')

# Adjust layout to prevent overlapping titles
plt.tight_layout()

# Show the plots
plt.show()




df['Education'].unique()


# making Marital_Status binary 
df['Marital_Status'] = df['Marital_Status'].replace( # replace rather than rename to transform column value and not column name
    {   
        "Married": "Couple",
        "Together": "Couple",
        "Absurd": "Alone",
        "Widow": "Alone",
        "YOLO": "Alone",
        "Divorced": "Alone",
        "Single": "Alone"
    })

#combining teen and kid as childhome
df['childhome'] = df[["Kidhome", "Teenhome"]].sum(axis=1)

#mixing childhome and Marital_status to just keep ppl number in household 
df['Family_Size'] = df['Marital_Status'].replace({"Alone": 1, "Couple": 2}) + df['childhome']

#renaming column we will keep for more clarity
df.rename(columns = {"MntWines" : "Wines", "MntMeatProducts" : "Meat"}, inplace=True)

#making Education simpler
df['Education'] = df['Education'].replace(
    {
        'Basic' : 'Undergraduate',
        '2n Cycle' : 'Undergraduate',
        'Graduation' : 'Graduate',
        'Master' :  'Postgraduate',
        'Phd' : 'Postgraduate'
    })

#combine all accepeted campaign and response since we don't care about specific
df['Campains-Accepted'] = df['AcceptedCmp1']+ df['AcceptedCmp2']+ df['AcceptedCmp3']+df['AcceptedCmp4']+df['AcceptedCmp5'] +df['Response']



#drop non useful column
df.drop(['Kidhome', 'Teenhome', 'Marital_Status', 'MntFruits', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 
    'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response', 'Dt_Customer', 'childhome', 'Complain'], axis=1, inplace=True)


#transform education into numerical values
from sklearn.calibration import LabelEncoder

Categorical_features = ['Education']
le = LabelEncoder()
for i in Categorical_features:
    df[i] = df[[i]].apply(le.fit_transform) # double '[[]]' to get a dataframe with one column rather than a series object




mask = ( df['Income'] <= 600000) & (df['Age'] <= 100) 
data = df[mask]


from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Min-Max Scaling (Normalization)
min_max_scaler = MinMaxScaler()
df_normalized = df.copy()  # Make a copy of the original DataFrame
df_normalized[df.columns] = min_max_scaler.fit_transform(df_normalized[df.columns])

# Standardization (Z-score Normalization)
standard_scaler = StandardScaler()
df_standardized = df.copy()  # Make a copy of the original DataFrame
df_standardized[df.columns] = standard_scaler.fit_transform(df_standardized[df.columns])



# Compute the correlation matrix
correlation_matrix = df.corr()

# Create the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='RdPu', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap of DataFrame')
plt.show()
