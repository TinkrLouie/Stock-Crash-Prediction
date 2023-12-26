import pandas as pd

df = pd.read_csv('crash_data.csv')

crash = df.drop(labels=['gvkey', 'ncskew_new_plus'], inplace=False, axis=1)
ncskew = df.drop(labels=['gvkey', 'crash_new320Plus'], inplace=False, axis=1)

crash.to_csv('crash.csv', index=False)
ncskew.to_csv('ncskew.csv', index=False)
