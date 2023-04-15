## DSAI-Project

### Brief report describing project
- A classification model through supervised learning to predict from 30 features of breast-cancer cells within per patient
whether they have Malignant (M) or Benign (B) tumor.
- 10 features describes the mean of different cell features, another 10 on its standard error, and last 10 on the "worst" or largest (mean
of the three largest values)
- A total of 6 different models were evaluated based on accuracy score and recall (TPR) for ranking purposes
- Dataset used is based only in US, Wisconsin data collected from University of Wisconsin

### Contributors
- Data preparation & cleaning (Ayub)
- Exploratory Data Analysis (Nahvin)
- Machine Learning Models & Evaluation (Manas)
- As a whole, the entire project structure and steps was discussed and agreed as a group

### Problem Definition
- How can we identify malignant or benign tumor in a patient having breast cancer?
- What tools and methods can we use to arrive at an accurate diagnosis?
- What are the implications of developing such models and what scope do they have in the future?

### Models Used
1. K-Nearest Neighbours
2. Random Forest
3. Decision Tree
4. Naive Bayes
5. Logistic Regression
6. Support Vector Machine

### Conclusion
- All 6 models did generally well with accuracy of over 93% for each
- Mild imbalance found in dataset did not require resampling/upweighting as models works and generalises well 
without the need to do so
- It was found that feature selection through using only 10 (the means of features) out of 30 features for prediction negatively affects model evaluation
- Logistic Regression model performs the best among others w/ 98.245% accuracy with 95.312% recall
- Other models fall behind by at most 5% in accuracy

### Learning points from this project
- "The model is just as good as the data it is trained on". 
Hence, improving the quality and quantity of the data will ultimately lead to better model performance
- Reinforce concepts on accuracy, recall, and other evaluation metrics
- Evaluating feature importance
- Learnt that normalising distributions of features does not necessarily improve model performance
