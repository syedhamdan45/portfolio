# Portfolio

## Personal Details:
**Name**: Syed Hamdan Mustafa

**Contact Number**: +1-647-444-4629

**Email**: syedhamdan45@gmail.com

**LinkedIn**: https://www.linkedin.com/in/syedhamdan45/

I am currently a Master of Engineering student at the University of Toronto pursuing the degree in Mechanical & Industrial Engineering with the emphasis in Data Analytics. I am seeking full-time work opportunity after I graduate in September 2020.

## Relevant Academic Courses:
- Foundations of Machine Learning and Data Analytics
- Introduction to Data Science and Analytics
- Decision Support Systems
- Big Data Science
- AI in Finance
- Financial Engineering

## Projects:

### Recommender System Project

The recommendation dataset used for the project is from a collection called MovieLens, which contains users' movie ratings and is popular for implementing and testing recommender systems. The specific dataset we will be using is the MovieLens 100K Dataset which contains 100,000 movie ratings from 943 users and a selection of 1682 movies.

The notebook explores three different similarity metrics in cosine, eucledian and jaccard. The algorithms used to make recommendation are as follows:
1. User Average
2. Popularity
3. Collaborative Filtering (User-User and Item-Item)
4. Probabilistic Matrix Factorization (PMF)

The different algorithms are evaluated by metrics such as RMSE, Precision@K, Recall@K after cross-validation. The Collaborative Filtering algorithm (Item-Item) is also used to make movie recommndations. What should you watch if you liked Men in Black? Only one way to find out.

Code: https://github.com/syedhamdan45/recommender-systems/blob/master/Recommender_Systems_Project.ipynb

### Fake News Detection

Natural Language Processing (NLP) Project is based on "Leaders Prize: Fact or Fake News?" Competition (http://leadersprize.truenorthwaterloo.com)

The project includes cleaning, exploration and visualiazation of data. In addition, the project implements various model such as:
1. Logistic Regression,
2. Random Forrest,
3. Naive Bayes and
4. Convulational Neural Network to classify the news headline as either Fake, True and Partly True.

Code: https://github.com/syedhamdan45/fake-news/blob/master/Group_19%20_Fake_News_Project.ipynb

Presentation: https://github.com/syedhamdan45/fake-news/blob/master/Fake%20News%20Model%20Presentation.pdf

Report: https://github.com/syedhamdan45/fake-news/blob/master/Fake%20News%20Model%20Consultation%20Report.pdf

### Reddit Classification Project

This project analyzes a sample of posts (without their comments) from a small subset of very active subreddits. The key features in this project are:
- Exploratory data analysis.
- Automatically classifying posts to their subreddit.
- Analyzing the sentiments in reddit post titles.

The EDA also includes data cleaning and understanding the importance of certain features which could be utilized in the classification of the posts. There are three algorithms implemented to do the classification:
1. Logistic Regression
2. Naive Bayes
3. Support Vector Machine

There are different feature encoding methods used as well as cross-validation and hyperparameter tuning. The algorithms are compared using the following metrics: accuracy, precision and recall. 

The final part of the project includes using the Vader Analyzier to compute the sentiment of the various posts.

Code: https://github.com/syedhamdan45/reddits-classification/blob/master/Reddit_Classification_Project.ipynb

### Hotel Reviews Sentiment Analysis Project

This project involves performing analysis of real hotel review data crawled from the Tripadvisor website to automatically identify positive and negative keywords and phrases associated with hotels and to better understand characteristics of data analysis tools, extracting explanatory review summaries, and human reviewing behavior. The data in this case is chosen for London (Ontario, Canada).

Trip Advisor Crawler code: https://github.com/aesuli/trip-advisor-crawler

Vader sentiment analyzer is used in this project to compute the sentiment for each hotel review. The importance of certain terms on the sentiment is determined by using Mutual Information. The reasons for the best and worst hotel reviews in the city are explored. The results are also visualized using various tools such as histograms, box plots and scatter plots.

Code: https://github.com/syedhamdan45/sentiment-analysis-hotels/blob/master/Hotel_Reviews_Sentiment_Analysis_Project.ipynb

### Information Retrieval System

The dataset used in the project includes test collection which is about 4,000 documents from US Government web sites and the topics/query are 15 needs for government information. Both were part of the TREC conference in 2003. The aim of the project is to run Whoosh on a large document collection for certain queries and approve on the baseline model.

Three softwares used:
- Whoosh, a pure-Python search engineering library, 
- NLTK, a natural language processing toolkit and 
- pytrec eval, an Information Retrieval evaluation tool for Python, based on the popular trec eval, the standard software for evaluating search engines with test collections.

Code: https://github.com/syedhamdan45/information-retrieval/blob/master/Information_Retrieval_Project.ipynb

### Canadian Federal Elections 2019 Sentiment Analysis

The goal of the project is to essentially use sentiment analysis on Twitter data to get insight into the 2019 Canadian elections.

The NLP project includes cleaning, exploration and visualization of data. The models are used in this project for two purposes:
- Predicting sentiment of the tweet related to each party (Liberals, Convervative, NDP)
- Classification of the tweets of negative sentiments to various categories

The features for the above are created using both Count Vectorizer and TF-IDF. The following models are implemented for the classificaiton:
1. Logisitic Regression
2. Naive Bayes
3. KNN
4. SVM
5. Decision Tree
6. Random Forest
7. XG Boost
8. Convolutional Neural Network

Code: https://github.com/syedhamdan45/canada-elections/blob/master/mustafa_1006193209_assignment2.ipynb

Presentation: https://github.com/syedhamdan45/canada-elections/blob/master/mustafa_1006193209_assignment2.pdf

### Data Scientists Salary Classification

The goal of the project is to identify skills and qualifications from the survey that are important to learn in data science field in order to have a high salary.
Dataset taken from: https://www.kaggle.com/kaggle/kaggle-survey-2018/

The model includes data cleaning, visualization and exploration. The data exploration includes feature engineering and selection to ensure the best features are selected. The model implemented here is Logistic Regression-One Versus Rest Classifier- since there this is an ordinary multi class classification algorithm multiclass problem. Bias vs Variance trade-off is also discussed in the code. Cross-validation and hyperparameter tuning is performed here too.

Code: https://github.com/syedhamdan45/classification-salary/blob/master/Salary_Classification_Project.ipynb

Presentation: https://github.com/syedhamdan45/classification-salary/blob/master/mustafa_1006193209_assignment1.pdf

### Network Analysis

Given a dataset of tweets, a hashtag is chosen (#vampirebite) and its network is explored and understood. The visualization of the network is done through NetworkX. The content is analyzed using the most common terms. The key players in the network are discovered using the following centrality measures:
1. Degree Centrality
2. Betweenness Centrality
3. Page Rank

The network connectivity patterns are also explored using cliques. A clique in an undirected graph is a subset of the nodes, such that every two different nodes are adjacent (directly connected with an edge).

Code: https://github.com/syedhamdan45/network-analysis/blob/master/Network_Analysis_Project.ipynb

### Linear Regression

Linear Regression model using different forms of gradient descent such as stochastic gradient descent and mini batch gradient descent on the classic Boston House dataset.

The code also uses two types of loss functions- Mean Squared Error Loss and Absolute Error loss. The code includes derivation of both types of loss functions and its implementation. The effect of learning rate is also explored.

Code: https://github.com/syedhamdan45/linear-regression/blob/master/Linear_regression.ipynb
