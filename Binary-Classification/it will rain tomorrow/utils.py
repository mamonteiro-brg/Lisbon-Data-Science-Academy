import numpy as np
import pandas as pd
from scipy.stats import randint

def basic_data_wrangling(dataframe):
	#Sub every empty postion with smtg
	_df = dataframe.copy()
	numericals = _df.select_dtypes(include=[np.number]).columns.tolist()

	for feature in numericals:
		_df[feature] = _df[feature].fillna(_df[feature].median())

	#Get categoricals
	categoricals = _df.select_dtypes(exclude=[np.number]).columns.tolist()    

	
	for feature in categoricals:
		_df[feature] = _df[feature].fillna(_df[feature].value_counts().idxmax())

	#Create dummies 
	_df = pd.get_dummies(_df, columns=categoricals, drop_first=True)
	
	return _df
	


random_search_parameter_space_dist = {
                   "max_depth": randint(1, 100),
                   "max_features": randint(1, len(independent_variables)),
                   "class_weight": ["balanced", None]
                  }
				  
randomized_search = RandomizedSearchCV(
                        estimator, 
                        random_search_parameter_space_dist,
                        cv=5, n_iter=250,
                        random_state=42,
                        return_train_score=True
)				 
				  