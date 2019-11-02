# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_daq as daq
import dash_html_components as html
from dash.dependencies import Input, Output

# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("intelligent_st.csv")
X = pd.DataFrame(df, columns=['fac1','fac2','fac3', 'fac4'])
Y=pd.DataFrame(df, columns=['score'])


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca_fit = pca.fit(X)
print(pca.explained_variance_ratio_)

principalComponents = pca.transform(X)
principalDf = pd.DataFrame(data=principalComponents, columns = ['principal component 1', 'principal component 2'])
Y=df['score']

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
X_case =pd.DataFrame({'fac1':[1],'fac2':[1],'fac3':[1], 'fac4':[1]})
Y_case=regressor.predict(X_case)
Y_case[0]

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server=app.server

app.layout = html.Div([
        
    html.H1('Predictor Pass or Fail'),
        
    html.Div([   
    html.Label('fac1'),
    dcc.Slider(id='gre-slider',
            min=0, max=1, step=0.2, value=0,
               marks={
        0: {'label': '0'},
        .20: {'label': '.20'},
        .40: {'label': '.40'},
        .60: {'label': '.60'},
        .80: {'label': '.80'},
        1.00: {'label': '1.00'},

                                            
    }),

html.Br(),
html.Label('fac2'),
dcc.Slider(id='toefl-slider',
            min=0, max=1, step=0.2, value=0,
               marks={
        0: {'label': '0'},
        0.2: {'label': '0.2'},
        0.4: {'label': '0.4'},
        0.6: {'label': '0.6'},
        0.8: {'label': '0.8'},
        1.0:{'label': '1.0'}
                                       
    }),

html.Br(),
html.Label('fac3'),
dcc.Slider(id='rating-slider',
            min=0, max=1, step=0.2, value=0.6,
               marks={
        0: {'label': '0'},
        0.2: {'label': '0.2'},
        0.4: {'label': '0.4'},
        0.6: {'label': '0.6'},
        0.8: {'label': '0.8'},
        1.0: {'label': '1.0'},
                                
    }),

html.Br(),
html.Label('fac4'),
dcc.Slider(id='sop-slider',
           min=0, max=1, step=0.2, value=0.2,
               marks={
        0: {'label': '0'},
        0.2: {'label': '0.2'},
        0.4: {'label': '0.4'},
        0.6: {'label': '0.6'},
        0.8: {'label': '0.8'},
        1.0: {'label': '1'},
                                    
    }),


],className="pretty_container four columns"),

  html.Div([ 

    daq.Gauge(
       id='my-gauge',
      showCurrentValue=True,
     color={"gradient":True,"ranges":{"red":[-10,-5],"yellow":[-5,0.20],"green":[.20,1.00]}},
    label="Probability",
   max=1,
   min=0,
   value=.50
   ),
])
    ])


@app.callback(
    Output('my-gauge', 'value'),
    [Input('gre-slider', 'value'),
     Input('toefl-slider', 'value'),
     Input('rating-slider', 'value'),
     Input('sop-slider', 'value')
    
     ])
def update_output_div(P1,
                      P2,
                      P3,
                      P4,
                     ):
   X_case =pd.DataFrame({'fac1':[P1],'fac2':[P2],'fac3':[P3],'fac4':[P4]})
   Y_case = regressor.predict(X_case)

   return Y_case[0]


if __name__ == '__main__':
    app.run_server()
