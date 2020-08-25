import pg8000
import sys
import pandas as pd

host = 'lifungprod.cctlwakofj4t.ap-southeast-1.redshift.amazonaws.com'
port = 5439
database = 'lifungdb'
user_name = 'btemymuser'
password = 'Lifung123'

conn = pg8000.connect(database=database, host=host,
                      port=port, user=user_name, password=password)
df = pd.read_sql_query('select * from r_bte_product_review_f', conn)
df.info(memory_usage="deep")
