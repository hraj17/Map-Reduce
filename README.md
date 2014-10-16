Map-Reduce
==========
Identifying Spam Accounts in Twitter

Social networking sites have become very popular in recent years and in an important part of our 
modern life. There are several kinds of social networking websites, among these, Twitter is fastest 
growing site. Twitter offers several free services to users which attracts many companies and 
spammers to profit out of advertisements and schemes. This pollutes the search results and may 
cause inconvenience to genuine users. Removing such spammers is important for the site's 
reputation. Hence Twitter bans the creation of serial or bulk accounts to “artificially inflate the 
popularity of users seeking to promote themselves on Twitter.” Identifying such accounts will 
provide advertisers with more accurate reports with respect to twitter usage and avoid users with 
reduced “annoying” tweets. 
For this project, we have used APIs provided by Twitter to collect live Twitter feed which we 
analyze and evaluate to label all the unique users that we have collected. Our project involves 
identifying Spam accounts after learning and creating a model for accounts which were identified 
by twitter as fake. We do this by using different classification algorithms which will be parallelized
to make them more efficient when used in Map Reduce Framework. The different classification 
algorithms used are k-Nearest Neighbors, Naive Bayesian with Binning, Decision Trees, and K-means classification algorithms.
