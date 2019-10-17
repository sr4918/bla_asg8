# -*- coding: utf-8 -*-
import os
os.getcwd()
os.chdir('C:/Users/samee/Dropbox/NYU-PhD/3. Fall 2019/Building Learning Applications/Assignment/Assignment 6')
import dash
import dash_core_components as dcc
import dash_daq as daq
import dash_html_components as html
from dash.dependencies import Input, Output

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("Student_Performance_Reshaped.csv")
X = pd.DataFrame(df, columns=['Paper 1','Paper 2','Paper 3','Paper 4','Paper 5','Paper 6'])
Y=pd.DataFrame(df, columns=['Paper 7'])


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca_fit = pca.fit(X)
print(pca.explained_variance_ratio_)

principalComponents = pca.transform(X)
principalDf = pd.DataFrame(data=principalComponents, columns = ['principal component 1', 'principal component 2'])
Y=df['Paper 7']

plt.scatter(x=principalDf['principal component 1'],y=principalDf['principal component 2'], c=Y, cmap="BuPu_r" )
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2 Dimensional representation of the data')
plt.rcParams['figure.figsize'] = [12, 10]
plt.show()


from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5)

regressor = LinearRegression()  
regressor.fit(X_train, Y_train) 

print("Intercept:",regressor.intercept_)

coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])  
coeff_df
Y_pred = regressor.predict(X_test)
X_case =pd.DataFrame({'Paper 1':[1],'Paper 2':[5],'Paper 3':[1],'Paper 4':[1],'Paper 5':[1],'Paper 6':[1]})
Y_case=regressor.predict(X_case)
Y_case[0]

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server=app.server

app.layout = html.Div([
        
    html.H1('Predictor Paper 7 Marks'),
        
    html.Div([   
    html.Label('Paper 1'),
    dcc.Slider(id='gre-slider',
            min=0, max=100, step=20, value=0,
               marks={
        0: {'label': '0'},
        20: {'label': '20'},
        40: {'label': '40'},
        60: {'label': '60'},
        80: {'label': '80'},
        100: {'label': '100'},

                                            
    }),

html.Br(),
html.Label('Paper 2'),
dcc.Slider(id='toefl-slider',
            min=0, max=100, step=20, value=10,
               marks={
        0: {'label': '0'},
        20: {'label': '20'},
        40: {'label': '40'},
        60: {'label': '60'},
        80: {'label': '80'},
        100: {'label': '100'},                               
    }),

html.Br(),
html.Label('Paper 3'),
dcc.Slider(id='rating-slider',
            min=0, max=100, step=20, value=20,
               marks={
        0: {'label': '0'},
        20: {'label': '20'},
        40: {'label': '40'},
        60: {'label': '60'},
        80: {'label': '80'},
        100: {'label': '100'},
                                
    }),

html.Br(),
html.Label('Paper 4'),
dcc.Slider(id='sop-slider',
           min=0, max=100, step=20, value=30,
               marks={
        0: {'label': '0'},
        20: {'label': '20'},
        40: {'label': '40'},
        60: {'label': '60'},
        80: {'label': '80'},
        100: {'label': '100'},
                                
    }),

html.Br(),
html.Label('Paper 5'),
dcc.Slider(id='lor-slider',
                   min=0, max=100, step=20, value=40,
               marks={
        0: {'label': '0'},
        20: {'label': '20'},
        40: {'label': '40'},
        60: {'label': '60'},
        80: {'label': '80'},
        100: {'label': '100'},
                                
    }),

html.Br(),
html.Label('Paper 6'),
dcc.Slider(id='gpa-slider',
            min=0, max=100, step=20, value=80,
               marks={
        0: {'label': '0'},
        20: {'label': '20'},
        40: {'label': '40'},
        60: {'label': '60'},
        80: {'label': '80'},
        100: {'label': '100'},
                                
    }),

],className="pretty_container four columns"),

  html.Div([ 

    daq.Gauge(
       id='my-gauge',
      showCurrentValue=True,
     color={"gradient":True,"ranges":{"red":[0,40],"yellow":[40,70],"green":[70,100]}},
    label="Probability",
   max=100,
   min=0,
   value=50
   ),
])
    ])


@app.callback(
    Output('my-gauge', 'value'),
    [Input('gre-slider', 'value'),
     Input('toefl-slider', 'value'),
     Input('rating-slider', 'value'),
     Input('sop-slider', 'value'),
     Input('lor-slider', 'value'),
     Input('gpa-slider', 'value')
     ])
def update_output_div(P1,
                      P2,
                      P3,
                      P4,
                      P5,
                      P6):
   X_case =pd.DataFrame({'Paper 1':[P1],'Paper 2':[P2],'Paper 3':[P3],'Paper 4':[P4],'Paper 5':[P5],'Paper 6':[P6]})
   Y_case = regressor.predict(X_case)

   return Y_case[0]


if __name__ == '__main__':
    app.run_server()
