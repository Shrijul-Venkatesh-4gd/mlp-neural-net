# mlp-neural-net
dataset : https://archive.ics.uci.edu/dataset/2/adult

dataset installation steps 
pip install ucimlrepo

from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
adult = fetch_ucirepo(id=2) 
  
# data (as pandas dataframes) 
X = adult.data.features 
y = adult.data.targets 
  
# metadata 
print(adult.metadata) 
  
# variable information 
print(adult.variables) 
