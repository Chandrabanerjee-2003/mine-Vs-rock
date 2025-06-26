# mine-Vs-rock
This is my first Machine Learning project!
The goal of this project is to classify whether a given sonar signal returns from a rock or a mine using supervised learning techniques.

Dataset Info
208 Samples
60 Numerical Features
1 Target Column:
R: Rock
M: Mine
Dataset Source: https://archive.ics.uci.edu/dataset/151/connectionist+bench+sonar+mines+vs+rocks

Technologies Used
Python
Numpy
Pandas
Scikit-learn
Google Colab

Workflow Summary
1. Import Libraries
Imported necessary libraries like NumPy, Pandas, and scikit-learn.

2. Load the Dataset
python
Copy
Edit
sonar_data = pd.read_csv('/content/Copy of sonar data.csv', header=None)
3. Exploratory Data Analysis
Viewed shape, head, and statistical summary of the dataset
Checked class distribution using value_counts()
Compared means of each feature grouped by label using groupby()

4. Preprocessing
Separated features (X) and labels (Y)
Split the data using train_test_split() with stratified sampling

5. Model Building
Used Logistic Regression:
model = LogisticRegression()
model.fit(x_train, y_train)

7. Model Evaluation
Training Accuracy: 83.42%
Test Accuracy: 76.19%

7. Prediction System
Created a prediction system to classify new sonar signal input:

if prediction[0]=='R':
    print('The object is a Rock')
else:
    print('The object is a Mine')
✅ Results
Dataset	Accuracy
Training	83.42%
Test	76.19%

