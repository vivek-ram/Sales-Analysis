import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from itertools import combinations
from collections import Counter
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.title('Sales Analysis')
st.text('Have a look througth the Dataset')
# loading the data frame
all_data = pd.read_csv("all_data.csv")
st.write(all_data)
# finding nan 
nan_df = all_data[all_data.isna().any(axis=1)]
# drop nan
all_data = all_data.dropna(how='all')
# Get rid of text in order date column
all_data = all_data[all_data['Order Date'].str[0:2]!='Or']
# Make columns correct type
all_data['Quantity Ordered'] = pd.to_numeric(all_data['Quantity Ordered'])
all_data['Price Each'] = pd.to_numeric(all_data['Price Each'])
# add month to column
all_data['Month'] = all_data['Order Date'].str[0:2]
all_data['Month'] = all_data['Month'].astype('int32')

all_data['Sales'] = all_data['Quantity Ordered'].astype('int') * all_data['Price Each'].astype('float')
# add city to column
def get_city(address):
    return address.split(",")[1].strip(" ")

def get_state(address):
    return address.split(",")[2].split(" ")[1]

all_data['City'] = all_data['Purchase Address'].apply(lambda x: f"{get_city(x)}  ({get_state(x)})")

st.title('Data Exploration!')

months = range(1,13)
fig = px.bar(all_data, x=months, y=all_data.groupby(['Month']).sum()['Sales'], labels=dict(x="Month number", y="Sales in USD ($)",width=800, height=400), title="What was the best month for sales? How much was earned that month?")
fig.update_layout(
    margin=dict(l=30, r=30, t=30, b=30)
)
fig.update_xaxes(type='category')

st.write(fig)
keys = [city for city, df in all_data.groupby(['City'])]

fig = px.bar(all_data, x=keys, y=all_data.groupby(['City']).sum()['Sales'], labels=dict(x="Cities", y="Sales in USD ($)",width=800, height=400), title="What city sold the most product?")
fig.update_layout(
    margin=dict(l=30, r=30, t=30, b=30)
)
st.write(fig)

# Add hour column
all_data['Hour'] = pd.to_datetime(all_data['Order Date']).dt.hour
all_data['Minute'] = pd.to_datetime(all_data['Order Date']).dt.minute
all_data['Count'] = 1
keys = [pair for pair, df in all_data.groupby(['Hour'])]
fig = px.line(all_data, x=keys, y=all_data.groupby(['Hour']).count()['Count'],labels=dict(x="Hours", y="Number of Sales",width=800, height=400))
fig.update_layout(
    title_text='What time should we display advertisements to maximize likelihood of customers buying product?'
)
fig.update_xaxes(type='category')
st.write(fig)

# https://stackoverflow.com/questions/43348194/pandas-select-rows-if-id-appear-several-time
df = all_data[all_data['Order ID'].duplicated(keep=False)]

# Referenced: https://stackoverflow.com/questions/27298178/concatenate-strings-from-several-rows-using-pandas-groupby
df['Grouped'] = df.groupby('Order ID')['Product'].transform(lambda x: ','.join(x))
df2 = df[['Order ID', 'Grouped']].drop_duplicates()

count = Counter()

for row in df2['Grouped']:
    row_list = row.split(',')
    count.update(Counter(combinations(row_list, 2)))


product_group = all_data.groupby('Product')
quantity_ordered = product_group.sum()['Quantity Ordered']

keys = [pair for pair, df in product_group]

fig = px.bar(x=keys, y=quantity_ordered,labels=dict(x="Products", y="Number of products sold",width=800, height=400))
fig.update_layout(
    title_text='What product sold the most? Why do you think it sold the most?', autosize=False,
    width=800,
    height=500
)
fig.update_xaxes(type='category')
st.write(fig)

prices = all_data.groupby('Product').mean()['Price Each']


# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(
    go.Bar(x=keys, y=quantity_ordered,name='product'),
    secondary_y=False,
)
fig.add_trace(
    go.Scatter(x=keys, y=prices, name="price"),
    secondary_y=True,
)

# Add figure title
fig.update_layout(
    autosize=False,
    width=800,
    height=600
)

# Set x-axis title
fig.update_xaxes(title_text="Products")

# Set y-axes titles
fig.update_yaxes(title_text="<b>Quantity Ordered</b>", secondary_y=False)
fig.update_yaxes(title_text="<b>Price ($)</b>", secondary_y=True)
st.write(fig)






