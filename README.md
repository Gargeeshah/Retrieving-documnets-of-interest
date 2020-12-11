# Retrieving documnets of interest using classification algorithm

## 1.INTRODUCTION
We have lots of observations, and we want to infer some kind of structure underlying these observations. 
Groups of related observations or clusters. We're gonna motivate everything  with a real world application, a case study i.e. about a task of retrieving documents of interest.
Information retrieval (IR) is the activity of obtaining  information system  resources that are relevant to an information need from a collection of those resources. Searches can be based on full-text or other content-based indexing. Information retrieval is the science of searching for information in a document, searching for documents themselves, and also searching for the metadata that describes data, and for databases of texts, images or sounds.

### 1.1  PROJECT  AIM
The retrieving data from wikipedia project will be helpful in information gathering from dataset with user friendly technique .Whatever data /information of user’s interest will be display on screen with the help of some data analysis algorithms (K-nearest  neighbour  method).
The main motive of project is to retrieve data of interest from Wikipedia.User will also get some suggestions to search information, related to his/her searched data ,by assigning weightage to suggested articles.The article which has low weightage will be nearest to the user’s data of interest ,and that article will have highest rank in the suggessions.This will be done by another concept called “TF-IDF”.With the help of data analysis algorithm”  we’re gonna  provide also the word count of each and every word in the article which going to be display to the user .

### 1.2 OBJECTIVES
1. The objective of Information Retrieval System is support of user search generation.
2. To  present the search results in a format that facilitates the user in determining relevant iems.
3. Historically data has been presented in an order dictated by how it was physically stored.Typically,this is in arrival to the system order,thereby always displaying the results of a search stored by time.For those users interested in current events this is useful.
4. This Information Retrieval System provide functions that provide the results of a query in order of  potential relevance to the user.
5. Even most sophisticated techniques use item clustering and link analysis to provide additional item selection insights.

### 1.3 ABSTRACT
Nowadays, the resources available on the web increases significantly. It then has a large volume of information, but without mastery of content. In this immense data warehouse research of current information retrieval systems do not allow users to obtain results to their requests that meet exactly their needs. This is due in large part to indexing techniques (key words, thesaurus). The result is that the user of the web wasting much of his time to examine a large number of Web page by searching for what he needs, because the Web does not provide service in this direction. The Semantic Web is the solution; this new vision of the web is to make web resources not only understandable by humans but also by machines. To improve the relevance of information retrieval, we propose in this paper an approach based on the use of domain ontology for indexing a collection of documents and the use of semantic links between documents in the collection to allow the inference of all relevant documents. The work involves the implementation of a system based on the use of OWL ontology for research pedagogical documents. In this case, the descriptors are not directly chosen in the documents but in the ontology and are indexed by concepts that reflect their meaning rather than words are often ambiguous. To perform a search based on meaning, documents and their descriptors are stored in OWL ontologies describing the documentary features of a document. The objective is to design two types of OWL ontologies: document ontology reserved for storage of all pedagogical documents and domain ontology reserved for well-structured of documents stored in the level of the document ontology and each document is indexed by its keywords and their synonyms.
	
## 2.REQUIREMENT  ANALYSIS

### 2.1 SOFTWARE  REQUIREMENT

1. The project “Retrieving data from Wikipedia” is implemented using Python. 
2. Python 2.7.14
3. Jupyter Notebook
4. Graphlab Library
5. TKinter python.
	
### 2.2  HARDWARE  REQUIREMENT
1. Processor: Intel dual core i3
2. Ram:4GB or more
3. Hard dsik :1 TB	

## 3. DESIGN  AND  MODELING
Here ,our training data for document retrieving task is going to be document id and document text and then we are gonna extract some set of features,the one that we will use here is our TF_IDF.

## 4.IMPLEMENTATION
### 4.1 K-nearest neighbour Algorithm:
KNN is a non-parametric, lazy learning algorithm. Its purpose is to use a database in which the data points are separated into several classes to predict the classification of a new sample point.

Breaking it Down – Pseudo Code of KNN
We can implement a KNN model by following the below steps:

1. Load the data
2. Initialise the value of k
3. For getting the predicted class, iterate from 1 to total number of training data points
     1. Calculate the distance between test data and each row of training data. Here we will use Euclidean distance as our distance metric since it’s the most popular method. The other metrics that can be used are Chebyshev, cosine, etc.
     2. Sort the calculated distances in ascending order based on distance values
     3. Get top k rows from the sorted array
     4. Get the most frequent class of these rows
     5. Return the predicted class

KNN makes predictions using the training dataset directly.

Predictions are made for a new instance (x) by searching through the entire training set for the K most similar instances (the neighbors) and summarizing the output variable for those K instances.
To determine which of the K instances in the training dataset are most similar to a new input a distance measure is used. For real-valued input variables, the most popular distance measure is Euclidean distance.
Euclidean distance is calculated as the square root of the sum of the squared differences between a new point (x) and an existing point (xi) across all input attributes j.
EuclideanDistance(x, xi) = sqrt( sum( (xj – xij)^2 ) )

KNN Algorithm is based on feature similarity: How closely out-of-sample features resemble our training set determines how we classify a given data point:

Example of k-NN classification. The test sample (inside circle) should be classified ether to the first class of blue squares or to the second class of red triangles. If k = 3 (outside circle) it is assigned to the second class because there are 2 triangles and only 1 square inside the inner circle. If, for example k = 5 it is assigned to the first class (3 squares vs. 2 triangles outside the outer circle).
	A few Applications and Examples of KNN
1.  Credit ratings — collecting financial characteristics vs. comparing people with similar financial features to a database. By the very nature of a credit rating, people who have similar financial details would be given similar credit ratings. Therefore, they would like to be able to use this existing database to predict a new customer’s credit rating, without having to perform all the calculations.
2.	Should the bank give a loan to an individual? Would an individual default on his or her loan? Is that person closer in characteristics to people who defaulted or did not default on their loans?
3.	In political science — classing a potential voter to a “will vote” or “will not vote”, or to “vote Democrat” or “vote Republican”.
4.	More advance examples could include handwriting detection (like OCR), image recognition and even video recognition.

Some pros and cons of KNN
Pros:
1. No assumptions about data — useful, for example, for nonlinear data
2. Simple algorithm — to explain and understand/interpret
3. High accuracy (relatively) — it is pretty high but not competitive in comparison to better supervised learning models
4. Versatile — useful for classification or regression
Cons:	
1. Computationally expensive — because the algorithm stores all of the training data
2. High memory requirement
3. Stores all (or almost all) of the training data
4. Prediction stage might be slow (with big N)
5. Sensitive to irrelevant features and the scale of the data

Quick summary of KNN
The algorithm can be summarized as:
1.	A positive integer k is specified, along with a new sample
2.	We select the k entries in our database which are closest to the new sample
3.	We find the most common classification of these entries
4.	This is the classification we give to the new sample
A few other features of KNN:	
1. KNN stores the entire training dataset which it uses as its representation.
2. KNN does not learn any model.
3. KNN makes predictions just-in-time by calculating the similarity between an input sample and each training instance.

### 4.2	TF_IDF (Term Frequency –Inverse Document Frequency)
In information retrieval, tf–idf or TFIDF, short for term frequency–inverse document frequency, is a numerical statistic that is intended to reflect how important a word is to a document in a collection . It is often used as a weighting factor in searches of information retrieval, text mining, and user modeling. The tf–idf value increases proportionally to the number of times a word appears in the document and is offset by the number of documents in the corpus that contain the word, which helps to adjust for the fact that some words appear more frequently in general. tf–idf is one of the most popular term-weighting schemes today; 83% of text-based recommender systems in digital libraries use tf–idf.

tf-idf Model for Page Ranking
tf-idf stands for Term frequency-inverse document frequency. The tf-idf weight is a weight often used in information retrieval and text mining. Variations of the tf-idf weighting scheme are often used by search engines in scoring and ranking a document’s relevance given a query. This weight is a statistical measure used to evaluate how important a word is to a document in a collection or corpus. The importance increases proportionally to the number of times a word appears in the document but is offset by the frequency of the word in the corpus (data-set).
How to Compute:
tf-idf is a weighting scheme that assigns each term in a document a weight based on its term frequency (tf) and inverse document frequency (idf). The terms with higher weight scores are considered to be more important.
Typically, the tf-idf weight is composed by two terms-	
1.	 Normalized Term Frequency (tf)
2.	Inverse Document Frequency (idf)

#### Step 1: Computing the Term Frequency(tf)
Frequency indicates the number of occurences of a particular term t in document d. Therefore,

tf(t, d) = N(t, d), wherein tf(t, d) = term frequency for a term t in document d.
N(t, d)  = number of times a term t occurs in document d

We can see that as a term appears more in the document it becomes more important, which is logical.We can use a vector to represent the document in bag of words model, since the ordering of terms is not important. There is an entry for each unique term in the document with the value being its term frequency.
Let us consider document:
Doc 2: Steve teaches at Brown University.
Given below are the terms and their frequency on  the document. [N(t, d)]

DOC2	STEVE	TEACHES	BROWN	UNIVERSITY
Tf	    1	    1	      1	     1

#### Step 2: Compute the Inverse Document Frequency – idf
It typically measures how important a term is. The main purpose of doing a search is to find out relevant documents matching the query. Since tf considers all terms equally important, thus, we can’t only use term frequencies to calculate the weight of a term in the document. However, it is known that certain terms, such as “is”, “of”, and “that”, may appear a lot of times but have little importance. Thus we need to weigh down the frequent terms while scaling up the rare ones. Logarithms helps us to solve this problem.
First of all, find the document frequency of a term t by counting the number of documents containing the term:

df(t) = N(t)
where-
df(t) = Document frequency of a term t
N(t) = Number of documents containing the term t

Term frequency is the occurrence count of a term in one particular document only; while document frequency is the number of different documents the term appears in, so it depends on the whole corpus. Now let’s look at the definition of inverse document frequency. The idf of a term is the number of documents in the corpus divided by the document frequency of a term.
idf(t) = N/ df(t) = N/N(t)
It’s expected that the more frequent term to be considered less important, but the factor (most probably integers) seems too harsh. Therefore, we take the logarithm (with base 2 ) of the inverse document frequencies. So, the idf of a term t becomes :
idf(t) = log(N/ df(t))	

Graphical User Interface:

 



## 7. CONCLUSION
We have successfully implemented machine learning algorithm, K-Nearest Neighbour in retrieving data from Wikipedia. Data analysis is performed as per the user data of interest. We have also predicted some sugessions to the user as per his/her searched data using TF_IDF concept of  giving weightage to the articles.
We have successfully developed Graphical user interface using TKinter in python for user friendly searching of data in this project.
