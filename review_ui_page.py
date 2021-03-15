import pickle as p
import pandas as pd
import numpy as np
import webbrowser as wb
import dash as d
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.express as px
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import plotly.graph_objs as plot

# creating a dash class instance
app = d.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
project_name='Sentiment Analysis with Insights'

def open_browser():
    wb.open_new("http://127.0.0.1:8050/")

# loading of model
def load_model():
    global scrappedReviews
    scrappedReviews = pd.read_csv(r'etsy_scrap_review.csv')
    scrappedReviews.columns=['reviews']
    scrappedReviews = scrappedReviews['reviews'].values.tolist()


    global pickle_model
    file = open("pickle_model.pkl", 'rb') 
    pickle_model = p.load(file)

    global vocab
    file = open("feature.pkl", 'rb') 
    vocab = p.load(file)

# checking the reviews using using count vectorizer and TFID Transformer
def check_review(reviewText):
    transformer = TfidfTransformer()
    loaded_vec = CountVectorizer(decode_error="replace",vocabulary=vocab)
    reviewText = transformer.fit_transform(loaded_vec.fit_transform([reviewText]))
    return pickle_model.predict(reviewText)

# pie chart creation using the balanced reviews csv file 
def pie():
    global pie_gr
    df1 = pd.read_csv(r'balanced_reviews.csv')
    labels = ['NEGATIVE','POSITIVE']
    df1 = df1[df1['overall'] != 3]
    df1['overall'] = np.where(df1['overall'] > 3,1,0)
    # labels = np.sort(df1['overall'].unique())
    df_cl_label = df1['overall'].value_counts().to_frame().sort_index()
    value_list = df_cl_label['overall'].tolist()

    trace = plot.Pie(labels=labels,
                   values=value_list,
                   marker=dict(colors=['#cc99ff', '#ff0000']))
    data = [trace]
    layout = plot.Layout(title='Balanced Reviews')
    pie_gr = plot.Figure(data=data, layout=layout)
    
# application's UI creation using DASH
def create_app_ui():
    global project_name
    main_layout = dbc.Container(
        dbc.Jumbotron(
                [
                    html.H1(id = 'heading', children = project_name, className = 'display-3 mb-4'),
                    html.Div(
                        className='mat-card', 
                        style={"display": "block", "margin": "15px"},
                        children=[html.H4(children='Sentiment Analysis Pie Chart'),
                                dcc.Graph(figure=pie_gr)]
                        ),
                    dbc.Container([
                        dcc.Dropdown(
                            id='dropdown',
                            placeholder = 'Select a Review',
                            options=[{'label': i[:100] + '....', 'value': i} for i in scrappedReviews],
                            value = scrappedReviews[0],
                            style = {'margin-bottom': '30px'}
                    )
                    ],
                    style = {'padding-left': '50px', 'padding-right': '50px'}
                    ),
                    dbc.Button("Submit", color="dark", className="mt-2 mb-3", id = 'button', style = {'width': '100px'}),
                    html.Div(id = 'result'),
                    dbc.Textarea(id = 'textarea', className="mb-3", placeholder="Enter the Review", value = 'Enter your text..', style = {'height': '150px'}),
                    dbc.Button("Submit", color="dark", className="mt-2 mb-4", id = 'button1', style = {'width': '100px'}),
                    html.Div(id = 'result1')
                    ],
                className = 'text-center'
                ),
            className = 'mt-4'
            )
    return main_layout

@app.callback(
    Output('result1', 'children'),
    [
    Input('button1', 'n_clicks')
    ],
    [
    State('textarea', 'value')
    ]
    )    

# updating the UI according to the reviews i.r Positive or Negative
def update_app_ui(n_clicks, textarea):
    result_list = check_review(textarea)
    
    if (result_list[0] == 0 ):
        return dbc.Alert("Negative", color="danger")
    elif (result_list[0] == 1 ):
        return dbc.Alert("Positive", color="primary")
    else:
        return dbc.Alert("Unknown Review!", color="dark")

@app.callback(
    Output('result', 'children'),
    [
    Input('button', 'n_clicks')
    ],
    [
     State('dropdown', 'value')
    ]
    )

# Dropdown updation function
def update_dropdown(n_clicks, value):
    result_list = check_review(value)
    
    if (result_list[0] == 0 ):
        return dbc.Alert("The above review is Negative!", color="danger")
    elif (result_list[0] == 1 ):
        return dbc.Alert("The above review is Positive!", color="primary")
    else:
        return dbc.Alert("Unknown", color="dark")

# Main function, assigning None to all the global variables used
def main():
    global app
    global project_name
    pie()
    load_model()
    open_browser()
    app.layout = create_app_ui()
    app.title = project_name
    app.run_server()
    app = None
    project_name = None

# calling the main function
if __name__ == '__main__':
    main()