# Module 4 Project - Online News Popularity

Authors - Aroa Gomez, Mohammed Hannan

#### Dataset:
 - Approximately 40,000 articles.
 - Articles parsed and key variables recorded.
 - Length of title, length of article, day published, positivitiy/negativity, subjectivity, channel etc.
 - Target variable: Number of times an article will be shared.
 
#### Files:
 - 'Mod 4 project notebook.ipynb'  contains the model definitions and outputs
 - 'OnlineNesPopularity.csv' contains full dataset
 - 'cleaning.py' contains data cleaning functions 
 - 'refactoring.py' contains modelling and cross validation refactorisation
 - 'OnlineNewsPopularity.pdf' contain presentation slides summarising the project

### Objective:

The objective of this project is to predict news articles popularity prior to its publication. The study will be applied on Mashable.com articles from 2015. The features of the articles are expressed in ratios and our target variable to explain 'popularity' is the number of times an article gets shared from the webstie. 
We will attempt a predictive model based on linear and polynomial regression. Cross validation will allow as to choose the best predictive model. 

### Target Audience:

Our client or target audience is a publication company or Mashable.com itself. The purpose and utility of this study is to allow them to decide beforehand what type of articles and what features those articles should have if their objective is to maximise the popularity of their publications
    
### Analysis: 

Once we finished exploration and cleaning of the data set we started with a Linear Regression approach to understand what variables could be excluded given that they did not provide any predictive power to our target variable. 
We used the logarithm of the number of shares and removed the repeated variables, for instance the max negative words and min negative words were removed and the average of negative words was used in the model. 
After a firs run of OLS we ruled out all the coefficients with p-value > 0.05 as they are not statistically significant. 
We then ran again an OLS and cross validated the training and testing performance. 
We did arrive at a very inconclusive result. Our model resulted in a very low R-squared value (0.10), meaning, we can only predict about 10% of the variance of our target. 
 
We therefore opted to define our variable with a polynomial regression to find if there was any overfitting characteristics. We landed on a very similar R-squared but with an improvement of 40%, (R-squared of 0.14)

Since our data was not fittable for the modelling techniques available at this point we decided to explore in more depth the descriptive values of our data set.

We observed a pattern on the channel and day of the week the article was published. 
Given the sample of articles collected we could determine the least popular channels or topics and the least suitable days of the week to publish news.

### Recommendations and Next Steps:
 
The conclusion of our analysis shows that social media and lifestyle articles are the ones with less shares and the weekends are the less desirable publication days if the objective is to maximise the popularity of the articles.

Our model did not explain the number of shares in any meaningful way. I couldn’t determine which of the features were more explicative as their coefficients were all equally close to each other and very close to zero. 

As a next step, we would like to recommend a logistic regression, which could potentially be more suitable for the objective of predicting if an article is popular or not. 

Gathering data on other news provider and publications onlines would also help to get a more general view on what articles get the most number of shares. 

It’s important to keep in mind that the purpose of this data set and the prediction is not trying to target a specific public or reader or understand the reader. This data analysis is purely based on the article features and language usage, which ultimately attempts to capture the quality of such articles and the sentiment and objectivity.
