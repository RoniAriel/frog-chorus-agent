import sqlite3
import pandas as pd


df = pd.read_csv("data/all_sites_glmm_second_order_time.csv")


df = df.rename(columns={
    "site_name": "recording_id",
    "frog_ID_i-previous": "frog_i",
    "frog_ID_j-previous": "frog_j",
    "frog_ID_k-current": "frog_k",
})


con = sqlite3.connect("data/frogs.db")
df.to_sql("transitions", con, index=False, if_exists="replace")


result = pd.read_sql_query("SELECT COUNT(*) AS n_rows FROM transitions", con)
print(result)

con.close()