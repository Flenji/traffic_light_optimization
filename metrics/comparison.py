# -*- coding: utf-8 -*-
"""
This script compares the different traffic light systems based on the 
xml output of the test simulations.

Authors: AAU CS (IT) 07 - 03
"""

import xmltodict
import os
import pandas as pd

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib




font = {'size'   : 18}

matplotlib.rc('font', **font)

foldername = "combined_flow"
filenames = [os.path.join(foldername,file) for file in os.listdir(foldername) if file.endswith(".xml")]

def fixed_first(li):
    res = []
    for el in li:
        if "fixed" in el:
            res.insert(0, el)
        else:
            res.append(el)
    return res

filenames = fixed_first(filenames)

def readXML(filename : str):
    """
    Reads an file into a string
    """
    result = ""
    f = open(filename, "r")
    for x in f:
        result += x
    f.close()
    return result

xml_strings = [readXML(filename) for filename in filenames]

xml_dicts = [xmltodict.parse(xml_string)["meandata"]["interval"] \
             for xml_string in xml_strings] #converts xml_strings to a python dictionary


def getAttributes(attr_list):
    """
    Converts the xml_dicts into one dictionary with the specified attributes.
    attr_list: List of attributes [str]
    """
    result = {}
    for idx,xml_dict in enumerate(xml_dicts):
        model = filenames[idx]
        result[model] = {}
        for attr_name in attr_list:
            result[model][attr_name] =[]
            for sub_dict in xml_dict:
                result[model][attr_name].append(sub_dict["edge"][attr_name])
    return result
            
data = getAttributes(["@speed","@traveltime","@waitingTime","@timeLoss","@density"])


#formatting the data for plotting
category = []
value = []
dataset = []
for k,v in data.items():
    
    for k_, v_ in v.items():
        k = k.split("\\")[-1].replace(".xml","")
        dataset.append(k)
        category.append(k_)
        num_v_ = []
        for el in v_:
            num_v_.append(float(el))
        value.append(np.mean(num_v_))

        
dd = {"Category":category, "Value":value, "Dataset":dataset}

df = pd.DataFrame(dd)
df["Normalized Value"]=df.groupby(["Category"])["Value"].transform(lambda x: x / x.max()) #scaling every value to 0<= x <= 1
df["Category"]=df["Category"].apply(lambda x: x.replace("@",""))

plt.figure(figsize =(10,6))
ax = sns.barplot(x='Category', y='Normalized Value', hue='Dataset', data=df)


ax.legend(loc='upper right')
plt.show()

fig = ax.get_figure()


fig.savefig("comparison.png")
#, bbox_to_anchor=(0.5, -0.2))



            


