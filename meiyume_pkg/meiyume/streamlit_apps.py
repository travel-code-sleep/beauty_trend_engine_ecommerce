import streamlit as st
import numpy as np 
import pandas as pd 
from IPython.display import display, HTML

st.title('Product')

"""
# My first app
Here's our first attempt at using data to create a table:
"""


new_df = pd.read_feather(r"C:\Amit\Meiyume\meiyume_data\sephora\metadata\no_cat_cleaned_sph_product_metadata_all_2019-10-31")
display(HTML(new_df.to_html(index=False)))

#option = st.sidebar.selectbox('select type', df['product_type'].unique())
#'You selected: ', option

#new_df = df[df.product_type==option]

option1 = st.sidebar.selectbox('select product', new_df['product_name'])
'You selected: ', option1


st.table(new_df[['low_p', 'high_p', 'mrp']][new_df.product_name == option1])
