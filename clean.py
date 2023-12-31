import pandas as pd
import numpy as np


df = pd.read_csv('crash_data.csv')

df.drop(labels=['gvkey'], inplace=True, axis=1)

#df.to_csv('ncskew.csv', index=False)

#def df_column_switch(df, column1, column2):
#    i = list(df.columns)
#    a, b = i.index(column1), i.index(column2)
#    i[b], i[a] = i[a], i[b]
#    df = df[i]
#    return df

#df = df_column_switch(df, 'crash_new320Plus', 'ncskew_new_plus')
#df.to_csv('crash.csv', index=False)

crash = df.drop(labels=['ncskew_new_plus'], inplace=False, axis=1)
ncskew = df.drop(labels=['crash_new320Plus'], inplace=False, axis=1)

crash.to_csv('crash.csv', index=False)
ncskew.to_csv('ncskew.csv', index=False)
