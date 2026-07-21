import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

data= pd.DataFrame({"Time":
["Day", "Day", "Day", "Day", "Day", "Day","Night", "Night", "Night", "Night", "Night", "Night"],'Season':
["Summer", "Summer", "Summer","Winter", "Winter", "Winter", "Summer", "Summer", "Summer","Winter", "Winter", "Winter"],'Day Type':
['Holiday','Working','Holiday','Holiday','Working','Working','Holiday','Working','Working','Holiday','Holiday','Working'],'Company':
["Crowd", "Crowd", "Alone","Crowd", "Alone","Crowd", "Crowd", "Crowd","Alone", "Crowd", "Alone","Crowd"], 'Familiarity':
["Familiar", "Familiar", "Familiar", "Familiar", "Familiar", "Unfamiliar", "Familiar", "Familiar", "Familiar", "Familiar", "Familiar","Unfamiliar"], 'Happy?':
["Happy", "Happy", "Happy", "Happy", "Unhappy", "Unhappy","Happy","Unhappy", "Unhappy", "Unhappy", "Unhappy", "Unhappy"]})

features=list(data.columns)[:-1]
target='Happy?'
#print(list(data[target].mode())[0])

def calculate_entropy(data,attr):
	prob = data[attr].value_counts(normalize=True)
	return -np.sum(prob*np.log2(prob))

def calculate_info_gain(data, split_attr, target_attr):
    total_ent = calculate_entropy(data,target_attr)
    weighted_ent = 0
    for value, subset in data.groupby(split_attr):
        weight = len(subset) / len(data)
        weighted_ent += weight * calculate_entropy(subset,target_attr)
    return total_ent - weighted_ent
    
def build_id3_tree(data, available_features, target_attr):

    if len(data[target_attr].unique()) == 1:
        return data[target_attr].iloc[0]
        
    if not available_features:
        return data[target_attr].mode()[0]
        
    gains = {feature: calculate_info_gain(data, feature, target_attr) for feature in available_features}
    best_feature = max(gains, key=gains.get)

    tree = {best_feature: {}}
    
    remaining_features = [f for f in available_features if f != best_feature]
    
    for value, subset in data.groupby(best_feature):
        tree[best_feature][value] = build_id3_tree(subset, remaining_features, target_attr)
            
    return tree

print("Building Decision Tree...\n")
decision_tree = build_id3_tree(data, features, target)
print("Tree structure:")
print(json.dumps(decision_tree,indent=2))

