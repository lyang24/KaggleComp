import numpy as np
import pandas as pd

#working on click train document
# display id # ad_id #clicked
X = pd.read_csv('C:\Users\Eric Yang\Desktop\Kaggle Comptetion\out Brain\clicks_train.csv')
y = X.groupby(['ad_id'])['clicked'].sum()
y = pd.Series(y)
z = X.drop_duplicates(subset = 'ad_id', keep = 'first')
z.sort(columns = 'ad_id', axis = 0, ascending = True, inplace = True)
z['total_clicks'] = y.values
#recycle variable y
y = X.groupby(['ad_id'])['ad_id'].count()
y = pd.Series(y)
z['ad_frequency'] = y.values
z.drop(['display_id', 'clicked'], axis = 1, inplace = True)
z['click_through_ratio'] = z['total_clicks']/z['ad_frequency']

#click through ratio complete

#now working on the document meta file
# publisher id # source id #document id #publish time
X2 = pd.read_csv('C:\Users\Eric Yang\Desktop\Kaggle Comptetion\out Brain\documents_meta.csv')
z = X.groupby(['publisher_id'])['source_id'].count()
# i tried to convert #publish time from str into time
X['publish_time'] = pd.to_datetime(X['publish_time'])
# out of bounds because the abnormalities below
y1 = X2.loc[X2['publish_time'].str[0] == '3']
y2 = X2.loc[X2['publish_time'].str[0] == '0']
y3 = X2.loc[(X2['publish_time'].str[0] == '0') & (X['publish_time'].str[1] != '0')]
# Not sure how to proceedX.