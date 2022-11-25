import pandas as pd 
import json

filename = "preprocessed_6.0E+09.json"
df = pd.read_json(filename , lines=True) 
df1 = df.T

list_of_dataframes = []
for row_idx in range(1,df1.shape[0]):
	label = df1.iloc[row_idx].name.split("/")[-1].split("_")[0]
	list_curr = df1.iloc[row_idx,0]
	
	df_curr = pd.DataFrame(list_curr)
	df_curr["label_subcategory"] = label
	list_of_dataframes.append(df_curr)
	
df_all = pd.concat(list_of_dataframes)
df_all.to_csv("analyze_binary_class.csv")

