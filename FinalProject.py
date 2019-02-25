
# coding: utf-8



# # Yelp Fusion API
# ## Chiara Maalouf

# ## Data Science on Business Information
# ### Specifically : Restaurants

# # Motivation
# ## Good Food = Happiness
# ## 1000 search request limit

# ## Who has better Food DC or Baltimore?


# In[41]:


import requests
import re
from bs4 import BeautifulSoup
import random
import pandas as pd
import json 
# YELPkey is a file where I store my API key
YELPkey ="bS9A3FTS5L6sDSZjvzQiVHP-MNvh84UqUBKiuJ7RhLYHcUy_Vy02RSb-v-sIwPIM6Ja9DeD8PHmF-C1AeJXYVD7S3nnWPwLuMOcBqrDYEvBijEGSBh8wSQXKBuMPXHYx"

import statistics


# ### Method: Request 1000 restaurants from each city

# In[57]:


# paramters for request
# location is required
# categories is optional
# the default limit is 20 and the max is 50
balti_params = {"location":"Baltimore","categories":"restaurants, All","limit":50}
dc_params = {"location":"Washington D.C.","categories":"restaurants, All","limit":50}


# In[43]:


# URL for the business search endpoint
url ="https://api.yelp.com/v3/businesses/search"


# In[58]:


# gets first 50 data points for baltimore restaurants
yelp_balti_restaurants = requests.get(url, headers={"Authorization":"Bearer "+YELPkey}, params=balti_params)

# gets first 50 data points for dc restaurants
yelp_dc_restaurants = requests.get(url, headers={"Authorization":"Bearer "+YELPkey}, params=dc_params)
 


# In[94]:


# makes first dataframe of the first 50 yelp results for Baltimore and DC
balti_restaurants1 = pd.DataFrame(yelp_balti_restaurants.json()['businesses'])

dc_restaurants1 = pd.DataFrame(yelp_dc_restaurants.json()['businesses'])


# In[96]:


index = 51
# to store all the baltimore data
balti_dataframes = [balti_restaurants1]
dc_dataframes = [dc_restaurants1]

# to get the rest of the baltimore restaurant data
while(index<952):
    # error checking to get the last 50 restaurants
    if(index==951):
        index= 950
    
    # Baltimore Restaurant Requests
    
    # makes request with new offset
    new_balti_params = {"location":"Baltimore","categories":"restaurants, All","limit":50}
    new_balti_params["offset"] = index
    new_restuarants_balti =requests.get(url, headers={"Authorization":"Bearer "+YELPkey}, params=new_balti_params)
    # makes dataframe of newly collected data
    new_balti_panda = pd.DataFrame(new_restuarants_balti.json()['businesses'])
    balti_dataframes.append(new_balti_panda)
    
    
    # DC Restaurant Requests
    
    # makes request with new offset
    new_dc_params = {"location":"Washington D.C.","categories":"restaurants, All","limit":50}
    new_dc_params["offset"] = index
    new_restuarants_dc =requests.get(url, headers={"Authorization":"Bearer "+YELPkey}, params=new_dc_params)
    # makes dataframe of newly collected data
    new_dc_panda = pd.DataFrame(new_restuarants_dc.json()['businesses'])
    dc_dataframes.append(new_dc_panda)
    
    index+=50

# makes dataframe of 1000 restaurants in Baltimore    
balti_restaurants = pd.concat(balti_dataframes, ignore_index=True)

# makes dataframe of 1000 restaurants in DC
dc_restaurants = pd.concat(dc_dataframes, ignore_index=True)
print(balti_restaurants['id'].count())
print(dc_restaurants['id'].count())


# In[99]:


print(balti_restaurants.describe())


# In[100]:


print(dc_restaurants.describe())


# # Are DC restuarant ratings significantly better than restuarant ratings in Baltimore?

# In[106]:


balti_ratings = balti_restaurants['rating']
dc_ratings = dc_restaurants['rating']

sub_balti_ratings_means = []
sub_dc_ratings_means = []


# In[434]:


# Inferential Stats Mathy Way
import statistics

# Using 2 stddevs for outliers
balti_ratings.sort_values()
dc_ratings.sort_values()

mean_balti_rating = statistics.mean(balti_ratings)
mean_dc_rating = statistics.mean(dc_ratings)

stddev_balit_rating = statistics.stdev(balti_ratings)
stddev_dc_rating = statistics.stdev(dc_ratings)



without_outliers_balti = []
without_outliers_dc = []

#for counting outliers
outlier_count_balti = 0
outlier_count_dc = 0

#creating bounds to find outliers
outlier_upperbound_balti = mean_balti_rating + 2*stddev_balit_rating
outlier_lowerbound_balti = mean_balti_rating - 2*stddev_balit_rating

outlier_upperbound_dc = mean_dc_rating + 2*stddev_dc_rating
outlier_lowerbound_dc = mean_dc_rating - 2*stddev_dc_rating
DAYS = 365
for i in range(1000):
    
    # 2015
    if(balti_ratings[i]< outlier_lowerbound_balti):
        outlier_count_balti +=1
    elif(balti_ratings[i]> outlier_upperbound_balti):
        outlier_count_balti+=1
    else:
        without_outliers_balti.append(balti_ratings[i])
        
    # 2017
    if(dc_ratings[i]< outlier_lowerbound_dc):
        outlier_count_dc +=1
    elif(dc_ratings[i]> outlier_upperbound_dc):
        outlier_count_dc+=1
    else:
        without_outliers_dc.append(dc_ratings[i])

        
mean_without_outliers_balti = statistics.mean(without_outliers_balti)
mean_without_outliers_dc = statistics.mean(without_outliers_dc)

    


# ### Discovering Outliers

# In[437]:


print("Baltimore: mean Restaurant ratings without outliers",mean_without_outliers_balti,sep=" : ")
print("DC: mean Restaurant ratings without outliers",mean_without_outliers_dc,sep=" : ")

print("Number of outliers in baltimore ratings", outlier_count_balti,sep=" : ")
print("Number of outliers in dc ratings", outlier_count_dc,sep=" : ")


# ### Inferential Statistics : Boot Strapping Confidence Intervals

# In[448]:


#Inferential Stats
# Bootstrap Confidence Intervals

import math

def std_error(stdev, n):
    return stdev/math.sqrt(n)

import random



for i in range(10000):
   
    for j in range(100):
        subsample_balti_ratings = []
        subsample_dc_ratings = []
        
        subsample_balti_ratings.append(balti_ratings[random.randint(0,len(balti_ratings)-1)])
        subsample_dc_ratings.append(dc_ratings[random.randint(0,len(dc_ratings)-1)])
        
    sub_balti_ratings_means.append(statistics.mean(subsample_balti_ratings))
    sub_dc_ratings_means.append(statistics.mean(subsample_dc_ratings))


# In[450]:


boot_mean_rating_balti = 0
boot_mean_rating_dc = 0

boot_stderr_rating_balti = 0
boot_stderr_rating_dc = 0

boot_CI_rating_balti = []
boot_CI_rating_dc = []

boot_mean_rating_balti = statistics.mean(sub_balti_ratings_means)
boot_mean_rating_dc = statistics.mean(sub_dc_ratings_means)

#calculates standard deviation of each bootstrap
boot_stdev_rating_balti = statistics.stdev(sub_balti_ratings_means)
boot_stdev_rating_dc = statistics.stdev(sub_dc_ratings_means)

#calculates standard errors of each bootstrap
boot_stderr_rating_balti = std_error(boot_stdev_rating_balti, 10000)
boot_stderr_rating_dc = std_error(boot_stdev_rating_dc, 10000)

#creating confidence interval

#first we order our means
sub_balti_ratings_means.sort()
sub_dc_ratings_means.sort()
# since we want 95% confidence we take 2.5% from each side
# 2.5% of 10,000 is 250 so the confidence interval is from the 250th element to the (10,000-251)th element
ninedyseven_five_perc = 10000-251
boot_CI_rating_balti = [sub_balti_ratings_means[251], sub_balti_ratings_means[ninedyseven_five_perc]]
boot_CI_rating_dc = [sub_dc_ratings_means[251], sub_dc_ratings_means[ninedyseven_five_perc]]


# In[451]:


print(" CI Baltimore Avg Restuarant Rating")
print(boot_CI_rating_balti) 
print(" CI DC Avg Restuarant Rating")
print(boot_CI_rating_dc)
print("mean Baltimore Rating = "+str(boot_mean_rating_balti))
print("mean DC Rating = "+str(boot_mean_rating_dc))


# ### Inferential Statistics 
# #### P Value Hypothesis:

# #### DC Restaurants have significantly higher average ratings than restaurants in Baltimore

# In[140]:


# Inferential Statistics pvalue hypothesis
def shuffleB(list_shuffle):
    for i in range(len(list_shuffle)-1):
        rand_index = random.randint(i,len(list_shuffle)-1)
        swap= list_shuffle[i]
        list_shuffle[i] =list_shuffle[rand_index]
        list_shuffle[rand_index] = swap
    return list_shuffle

statistic_pool=[]
ratings_balti_dc =[]

data_points = 1000

# combines daily calls of both years
for i in range(data_points):
    ratings_balti_dc.append(balti_ratings[i])
    ratings_balti_dc.append(dc_ratings[i])
    
# creates 1000 "fake universes" to test hypothesis
for i in range(10000):
    shuffled_stats_pool = shuffleB(ratings_balti_dc)
    shuffled_Bmore_stat_pool = shuffled_stats_pool[:len(shuffled_stats_pool)//2]
    shuffled_DC_stat_pool = shuffled_stats_pool[len(shuffled_stats_pool)//2:]
    
    # get difference of means value of newly made "fake universe" 2015-2017
    difference_means = statistics.mean(shuffled_Bmore_stat_pool) -statistics.mean(shuffled_DC_stat_pool)
    statistic_pool.append(difference_means)
    #test_universe_15_means.append(statistics.mean(shuffled_15_stat_pool))
    #test_universe_17_means.append(statistics.mean(shuffled_17_stat_pool))


# In[457]:


p=0
# finds where regular mean value is amoung the "fake universe" means
actual_balti_dc_difference = mean_dc_rating- mean_balti_rating
#mean_balti_rating - mean_dc_rating

statistic_pool.sort()

def p_value(stat_pool, norm_stat):
    count = 0
    for diff in range(len(stat_pool)):
        if stat_pool[diff] >abs(norm_stat): 
            count+=1
    return count/len(stat_pool)
        


# In[458]:


# gets p-value for null hypothesis
p = p_value(statistic_pool, actual_balti_dc_difference)

print(p)


# Since the p-value (.0007) is less than 1%, it is statistically significant.
# 
# I am 99% certain that the average DC restaurant ratings  are significantly better than the average Baltimore restaurant ratings

# In[407]:


lat = []
long = []
for i in range(all_bmore_info.shape[0]):
    lat.append(all_bmore_info['coordinates.latitude'])
    long.append(all_bmore_info['coordinates.longitude'])
                    
                        


# In[408]:


Latitude = pd.Series(v  for v in lat) 


# In[409]:


Longitude = pd.Series(v  for v in long) 


# In[459]:


# all_bmore_info.dropna(subset=['coordinates.latitude', 'coordinates.longitude'])
# all_bmore_info.dropna(axis=0)
# import matplotlib.pyplot as plt
# heatmap_balti = plt.hist2d(all_bmore_info['coordinates.longitude'], all_bmore_info['coordinates.latitude'], bins=100,  );
# plt.title("Number of Restaurants in Baltimore")
# plt.ylabel("Latitude");
# plt.xlabel("Longitude");
# plt.legend= "true";


# In[143]:


import sklearn
import sklearn.cluster
import sklearn.linear_model
import sklearn.preprocessing
import random

get_ipython().run_line_magic('matplotlib', 'inline')


# In[260]:


import json

import pandas as pd 
from pandas.io.json import json_normalize

# all 1000 id's of baltimore restaurants
ids =balti_restaurants['id']
more_info_url ="https://api.yelp.com/v3/businesses/"


# ## For every Baltimore restuarant available to me, I looked up their business info using the business info endpoint.

# In[255]:


# all 1000 id's of baltimore restaurants
business_info = []
ids =balti_restaurants['id']
for i in range(ids.count()):
    new_url = more_info_url+ids[i]
    new_restuarants_info =requests.get(new_url, headers={"Authorization":"Bearer "+YELPkey})
    # makes dataframe of newly collected data
    new_info_panda = json_normalize(new_restuarants_info.json())
    business_info.append(new_info_panda)
    
    


# In[381]:


all_bmore_info = pd.concat(business_info, ignore_index=True)


# In[414]:


import matplotlib.pyplot as plt
#heatmap_balti = plt.hist2d(all_bmore_info['coordinates.longitude'], all_bmore_info['coordinates.latitude'], bins=100,  );
#plt.title("Number of Permits Issued in Baltimore")
#plt.ylabel("Latitude");
#plt.xlabel("Longitude");
#plt.legend= "true";


# In[460]:


print(all_bmore_info.corr())


# In[ ]:





# In[ ]:





# #### Starts Creating sets for Logistics Regression

# In[417]:


train_set = pd.DataFrame()
test_set = pd.DataFrame()
# YOUR CODE HERE
# 30% of Dataframe goes to testing
# 70% is for training
row_count = all_bmore_info.shape[0]
shuffled_bmore = all_bmore_info.sample(n=row_count)
thirty_percent =(int)( row_count*(0.3));

test_set, train_set = shuffled_bmore[:thirty_percent], shuffled_bmore[thirty_percent:] 

print(train_set.shape[0])
print(test_set.shape[0])


# # Logistic Regression: Can We predict if a Restaurant 'is claimed' by the Number of Reviews a Restaurant has?

# #### Created a  Binary Column if the Restaurant is Claimed or not

# In[430]:


#test_binary = pd.Series()
#train_binary = pd.Series()

# YOUR CODE HERE


train_set["Binary_Claimed"] = train_set["is_claimed"].apply(lambda x : 0 if x else 1)
test_set["Binary_Claimed"] = test_set["is_claimed"].apply(lambda x : 0 if x else 1)

test_binary = test_set["Binary_Claimed"].tolist()
train_binary = train_set["Binary_Claimed"].tolist()


# In[461]:


logistic_model=0


feature_train = train_set[["review_count"]]
outstanding_train = train_set[["Binary_Claimed"]]

feature_test = test_set[["review_count"]]
outstanding_test = test_set[["Binary_Claimed"]]
logistic_model = sklearn.linear_model.LogisticRegression().fit(feature_train,outstanding_train)
log_coeff = logistic_model.coef_
log_intercept = logistic_model.intercept_


# In[ ]:


print(log_coeff)
print(log_intercept)


# # Logistic model for predicting if a business is claimed :

# is_claimed = -0.00703223*review_count -1.12585481

# In[462]:


y_prime = logistic_model.predict(feature_test)

print("Scores: ")
print("\tAccuracy: ", sklearn.metrics.accuracy_score(outstanding_test,y_prime))
print("\tPrecision: ", sklearn.metrics.precision_score(outstanding_test,y_prime))
print("\tRecall: ", sklearn.metrics.recall_score(outstanding_test,y_prime))
print("\tF1: ", sklearn.metrics.f1_score(outstanding_test,y_prime))


# In[466]:


import sklearn.dummy


# Finding Scores for Random
random_outstanding_train = train_set["Binary_Claimed"].sample(n=train_set.shape[0])
random_outstanding_test = test_set["Binary_Claimed"]
random_logistic_model = sklearn.dummy.DummyClassifier(strategy='uniform').fit(feature_train, random_outstanding_train)
#random_logistic_model = sklearn.linear_model.LogisticRegression().fit(feature_train,random_outstanding_train)
random_prime = random_logistic_model.predict(feature_test)



# Finding Scores for Most Common Baseline
train_set["Most Common Test"] =train_set["Binary_Claimed"].apply(lambda x :0)
test_set["Most Common Test"]= test_set["Binary_Claimed"].apply(lambda x: 0)

mcommon_outstanding_train = train_set["Most Common Test"]
mcommon_outstanding_test = test_set["Most Common Test"]
mcommon_logistic_model = sklearn.dummy.DummyClassifier(strategy='most_frequent').fit(feature_train,mcommon_outstanding_train)
#mcommon_logistic_model = sklearn.linear_model.LogisticRegression().fit(feature_train,mcommon_outstanding_train)
mcommon_prime = mcommon_logistic_model.predict(feature_test)



# In[464]:


print("Random Scores: ")
print("\tAccuracy: ", sklearn.metrics.accuracy_score(outstanding_test,random_prime))
print("\tPrecision: ", sklearn.metrics.precision_score(outstanding_test,random_prime))
print("\tRecall: ", sklearn.metrics.recall_score(outstanding_test,random_prime))
print("\tF1: ", sklearn.metrics.f1_score(outstanding_test,random_prime))


# In[465]:


print("Most Common Scores: ")
print("\tAccuracy: ", sklearn.metrics.accuracy_score(outstanding_test,mcommon_prime))
print("\tPrecision: ", sklearn.metrics.precision_score(outstanding_test,mcommon_prime))
print("\tRecall: ", sklearn.metrics.recall_score(outstanding_test,mcommon_prime))
print("\tF1: ", sklearn.metrics.f1_score(outstanding_test,mcommon_prime))


# In[ ]:





# # What can we learn by Clustering By Review Count Rating and Price?

# ### Can we make clusters that describe the price range?

# Price is one of 4 options $, $$, $$$, or $$$$

# ### To clean the data I created a new colllumn that quantified the price on  a scale of 1-4

# In[383]:


# cleaning data
all_bmore_info.dropna(subset=['price'], inplace=True)
print(all_bmore_info['price'].count())


# In[384]:


all_bmore_info['Numerical_1'] = all_bmore_info["price"].apply(lambda x:1 if x=='$' else 0)
all_bmore_info['Numerical_2'] = all_bmore_info["price"].apply(lambda x:2 if x=='$$' else 0)
all_bmore_info['Numerical_3'] = all_bmore_info["price"].apply(lambda x:3 if x=='$$$' else 0)
all_bmore_info['Numerical_4'] = all_bmore_info["price"].apply(lambda x:4 if x=='$$$$' else 0)


# In[385]:


all_bmore_info['Numerical_Price'] = all_bmore_info['Numerical_1'] + all_bmore_info['Numerical_2']+ all_bmore_info['Numerical_3'] + all_bmore_info['Numerical_4']


# In[386]:


all_bmore_info.corr()


# In[343]:


categories = []
for i in range(all_bmore_info['categories'].count()):
    categories.append(all_bmore_info['categories'][0][0]['alias'])
    
categories_series = pd.Series(v  for v in categories) 


# In[387]:


#print(categories_series)


# In[264]:


row_count = all_bmore_info.shape[0]
shuffled_restaurants = all_bmore_info.sample(n=row_count)


# In[375]:


scaled_restaurants = 0
cluster_data = all_bmore_info[['review_count','Numerical_Price','rating']]

scaleData = pd.DataFrame(sklearn.preprocessing.MinMaxScaler().fit_transform(cluster_data),index=all_bmore_info['alias'], columns=cluster_data.columns)
#print(scaleData)
scaleData.dropna;


# In[376]:


clusters = sklearn.cluster.KMeans().fit(scaleData)
clusters.labels_
clusters.score(scaleData)
numclusters=[]
for num in range(1,21):
    numclusters.append(sklearn.cluster.KMeans(n_clusters=num).fit(scaleData).score(scaleData))
matplotlib.pyplot.plot(numclusters);


# In[377]:


all_bmore_info['cluster_num'] = pd.Series(range(len(all_bmore_info)))
cluster_data['labels'] = sklearn.cluster.KMeans(n_clusters=5).fit(scaleData).labels_


# In[380]:


cluster_data.groupby('labels').mean()


# #### 0 : well reviewed moderately expensive restaurants with decent food
# #### 1 :  not well reviewed cheap restaurants with good food
# #### 2 : not well reviewed cheap restaurants with decent food
# #### 3 : well reviewed moderately expensive restaurants with good food
# #### 4 : well reviewed expensive restaurants with good food
# 
# 
# 
# 

# In[370]:


#print(cluster_data)
cluster_data[cluster_data.labels == 0].describe()
#print(all_bmore_info['alias'])


# # Conclusions
# ## YELP API = not so good for data science
# ## Numerical data limited
# ## DC restaurants > Baltimore

# In[371]:


cluster_data[cluster_data.labels == 1].describe()


# In[372]:


cluster_data[cluster_data.labels == 2].describe()


# In[373]:


cluster_data[cluster_data.labels == 3].describe()


# In[467]:


cluster_data[cluster_data.labels == 4].describe()


# In[472]:


cluster_data.plot.scatter('rating','Numerical_Price',c='labels',cmap='rainbow',title='Price by Rating', colorbar=False);
#cluster_data.plot.scatter('review_count','Numerical_Price',c='labels',cmap='rainbow',title='Price by Review count', colorbar=False);


# In[442]:


# Graphing ratings og baltimore restaurants
import matplotlib as plt
##avg_per_YM = balti_restaurants.groupby('review_count','ratings')

#baltimore_ratings_by_revies = balti_restaurants.plot.scatter(['review_count'],['rating'])#avg_per_YM.plot( title="ratings by review count");


baltimore_ = balti_restaurants.plot.scatter(['rating'],['review_count'],title='Baltimore')


dc_ = dc_restaurants.plot.scatter(['rating'],['review_count'], title='DC')

