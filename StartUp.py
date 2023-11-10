import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
import seaborn as sns

data = pd.read_csv('startUp(1).csv')
print(data.head())

dx = data.copy()
dx.isnull().sum()

# SCALING THE DATA
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

for i in dx.columns:
  if dx[i].dtypes != 'O':
    dx[[i]] = scaler.fit_transform(dx[[i]])

dx.head()

# ENCODING THE DATA
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

for i in dx.columns:
  if dx[i].dtypes == 'O':
    dx[i] = encoder.fit_transform(dx[i])

dx.head()

def plotter(dataframe,a,b, dependent):
    sns.set(style = 'darkgrid')
    plt.figure(figsize = (18, 3))

    plt.subplot(1, 3, 1)
    sns.regplot(x = dataframe[a], y = dataframe[dependent], ci = 0)
    plt.title(f"Correlation Btw {a} and {dependent} is: {dataframe[dependent].corr(dataframe[a]).round(2)}")

    plt.subplot(1, 3, 2)
    sns.regplot(x = dataframe[b], y = dataframe[dependent], ci = 0)
    plt.title(f"Correlation Btw {b} and {dependent} is: {dataframe[dependent].corr(dataframe[b]).round(2)}")

    plt.show()

plotter(dx, 'R&D Spend', 'Administration', 'Profit')
plotter(dx,  'Marketing Spend', 'State', 'Profit')

# drop State since it doesn't satisfy the assumption of Linearity
dx.drop('State', axis = 1, inplace = True)
# dx.columns

# Assumption of Multicolinearity
plt.figure(figsize = (9, 3))
sns.heatmap(dx.corr(), annot = True, cmap = 'BuPu')

# Train And Test Split
x = dx.drop('Profit', axis = 1)
y = dx.Profit

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size = 0.80, random_state = 45)
print(f'xtrain: {xtrain.shape}')
print(f'ytrain: {ytrain.shape}')
print(f'xtest: {xtest.shape}')
print(f'ytest: {ytest.shape}')

train_set = pd.concat([xtrain, ytrain], axis = 1)
test_set = pd.concat([xtest, ytest], axis = 1)

print(f'\t\tTrain DataSet')
print(train_set.head())
print(f'\n\t\tTest DataSet')
print(test_set.head())

# --------- Modelling ----------
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

lin_reg = LinearRegression()

lin_reg.fit(xtrain, ytrain) # --------------------------Create a linear regression model

# -------------- cross validation -------------
cross_validate = lin_reg.predict(xtrain)
score = r2_score(cross_validate, ytrain)
print(f'The Cross Validation Score is: {score.round(2)}')

# Model Metrics and Testing
test_prediction = lin_reg.predict(xtest)
score = r2_score(test_prediction, ytest)
print(f'The Cross Validation Score is: {score.round(2)}')

import pickle

# save model
pickle.dump(lin_reg, open('StartUp_Model.pkl', "wb"))

print(f'\nModel Is Saved')

# ----------------- STREAMLIT DEVELOPMENT --------------------

st.markdown("<h1 style = 'color: #BEADFA; text-align: center; font-family: Helvetica, sans-serif '>START UP PROJECT</h1>", unsafe_allow_html = True)

st.markdown("<h4 style = 'margin: -25px; color: #610C9F; text-align: center; font-family: script'>Built By Orpheus</h4>", unsafe_allow_html = True)

st.image('pngwing.com.png', width = 600, )

st.markdown("<h2 style = 'color: #363062; text-align: center; font-family: montserrat'>Background Of Study</h2>", unsafe_allow_html = True)

# st.markdown('<br><br>', unsafe_allow_html = True)

st.markdown("<p>By analyzing a diverse set of parameters, including Market Expense and Research and Development Spending, competitive landscape, financial indicators, and operational strategies, our team seeks to develop a robust predictive model that can offer valuable insights into the future financial performance of startups. This initiative not only empowers investors and stakeholders to make data-driven decisions but also provides aspiring entrepreneurs with a comprehensive framework to evaluate the viability of their business models and refine their strategies for long-term success.</p>", unsafe_allow_html = True)

st.sidebar.image('pngwing.com (1).png')

data = pd.read_csv('startUp(1).csv')
st.write(data.head())
st.sidebar.markdown('<br><br>', unsafe_allow_html= True)

# Select Your Preferred Input Style
input_type = st.sidebar.radio("Select Your Preferred Input Style", ["Slider", "Number Input"])
if input_type == 'Slider':
    st.sidebar.header('Input Your Information')
    research = st.sidebar.slider("R&D Spend", data['R&D Spend'].min(), data['R&D Spend'].max())
    admin = st.sidebar.slider("Administration", data['Administration'].min(), data['Administration'].max())
    mkt_spend = st.sidebar.slider("Marketing Spend", data['Marketing Spend'].min(), data['Marketing Spend'].max())
else:
    st.sidebar.header('Input Your Information')
    research = st.sidebar.number_input("R&D Spend", data['R&D Spend'].min(), data['R&D Spend'].max())
    admin = st.sidebar.number_input("Administration", data['Administration'].min(), data['Administration'].max())
    mkt_spend = st.sidebar.number_input("Marketing Spend", data['Marketing Spend'].min(), data['Marketing Spend'].max())

st.header('Input Values')
#Bring all the input into a dataframe
input_variable = pd.DataFrame([{'R&D Spend':research, 'Administration': admin, 'Marketing Spend': mkt_spend}])
st.write(input_variable)

# Standard Scale the Input Variable
from sklearn.preprocessing import StandardScaler

for i in input_variable.columns:
    input_variable[i] = StandardScaler().fit_transform(input_variable[[i]])

# Load model
import pickle
model = pickle.load(open('startUpModel.pkl', "rb"))

tab1, tab2 = st.tabs(["Modelling", "Interpretation"])
with tab1:
    if st.button('Press To Predict'):
        model.predict(input_variable)
        st.toast('Profitability Predicted')
        st.image('pngwing.com (2).png', width = 200)
        st.success('Predicted. Pls check the Interpretation Tab for interpretation')

with tab2:
    st.subheader('Model Interpretation')
    st.write(f"Profit = {model.intercept_.round(2)} + {model.coef_[0].round(2)} R&D Spend + {model.coef_[1].round(2)} Administration + {model.coef_[2].round(2)} Marketing Spend")

    st.markdown("<br>", unsafe_allow_html= True)

    st.markdown(f"- The expected Profit for a startup is {model.intercept_}")

    st.markdown(f"- For every additional 1 dollar spent on R&D Spend, the expected profit is expected to increase by ${model.coef_[0].round(2)}  ")

    st.markdown(f"- For every additional 1 dollar spent on Administration Expense, the expected profit is expected to decrease by ${model.coef_[1].round(2)}  ")

    st.markdown(f"- For every additional 1 dollar spent on Marketting Expense, the expected profit is expected to increase by ${model.coef_[2].round(2)}")

