
---
---

<br>

# Step-1: (Understand Your Data):

- `Ask some basic question.`

- `EDA -> Univariate Analysis.`

- `EDA -> Bi-variate Analysis` 

- `EDA -> Multivariate Analysis.`

- `Using Pandas Profiler.`


<br>

### Ask Some Basic Question: 

- How does the data look like : `df.head() best df.sample()`

- How big is the data : `df.shape`

- Are there are any missing value : `df.isnull().sum()`

- Are there any duplicate value : `df.duplicated().sum()`

- What is the data type of columns + memory use : `df.info()`

- How does the data look like mathematically : `df.describe()`

- How is the correlation between columns : `df_corr.corr()['Survived']` df_corr(only_for numerical column)

<br>

### Univariate Analysis:

- **1) First we focus on categorical data :**

    `Note1: কিছু কিছু Numerical Column, Categorical হিসেবে আমাদের treat করতে হবে । যেমনঃ class column যেখানেঃ 1,2,3 দিয়ে দেওয়া অর্থাৎ, 1st class, 2nd class, 3rd class  । Numerical Column গুলো অনেক বেশি distributed হবে যেমনঃ মানুষের বয়স ১ থেকে ১৫০ এর মধ্যে হতে পারে, এখানে, ভ্যালু ২-৩ টা এর মধ্যে সীমাবন্দ থাকতেছে না । `

    `Note2: অনেক সময় Numerical Column কে  Categorical এ convert করলে model এর performance  increase হয় ।  `
    - Analysis data with  : ` Countplot:`
    - Analysis data with  : ` Pie Chart`
    
- **2) Then, in numerical data :**

    - Analysis data with  : ` Histrogram `

    - Analysis data with  : ` Kde Plot `

    - Analysis data with  : ` Box Plot `

    - Find the skewness   : `pd[""].skew()`

<br>

### Multivariate Analysis:

- Scatter Plot: `(Numerical - Numerical)`

- Bar plot: `(Numerical - Categorical)`

- Box Plot `(Numerical-Categorical)`

- Kde Plot `(Numerical - Categorical)`

- Heatmap `(Categorical-Categorical)`


---
---

<br>

# Step-2 : (Handilling Missing Values):

**i)** Remove them `CCA(Complete Case Analysis)`

**ii)** Impute them
- Univariate `(Simple Imputer Class)`
    - Numerical 
        - Mean
        - Median 
        - Arbitrary Value Imputation
        - End of Distribution Imputation
        - Random value Imputation (lec:38)
    - Categorical
        - Mode or Most-Frequent (lec:37)
        - Missing Indicator (lec:38)
- Multivariate 
    - KNN Imputer
    - Iterative Imputer (base on MICE Algorithrm)

**iii) Automatically select value for Imputation:** 
- gridSearchCV

**NOTE:**

`MCAR(Missing Completly At Random: আমরা যখন data collect করেছি তখন কয়েকটা row এর data collect  হয়নি ।)    MAR(Missing At Random: আমাদের কাছে ৫টা column আছে এর যেকোন একটা row তে একটা column এর value missing ।)    MNAR(Missing Not At Random: আমরা জেনে বুঝে data remove করেছি । )`


## **1. CCA :** 
- Delte the whole row.
- MCAR (Missing Completely At Random).
- Less than 5% data is missing (Appling condition).
#### Advantage:
- Easy To Implement.
- Preserve Variable Distribution (If Data is MCAR).
#### Disadvantage:
- It can exclude a large fraction of the original dataset (if missing data is abundant).
- Excluded observations could be informative for the analysis (if data is not missing at random).
- When using our models in production, the model will not know how to handle missing data.

`প্রথমে আমরা missing value এর % বের করে যেই column গুলোতে 5% এর নিচে missing value আছে সেই missing value column গুলোকে col নামে save করলাম । এর পরে দেখবো, সব গুলো কলামের উপর CCA প্রয়োগ করলে আমাদের কাছে কত% remaining data থাকবে। এরপর দেখবো Preserve Variable Distribution থাকে কি না(with kdeplot) । Categorical data এর ক্ষেত্রে দেখবো, আগে ও পরে percentange of unique category almost same কি না । `

## **2. Mean And Median:**

- If data is MCAR.
- If Distribution is Normal then we use mean.
- If Distribution is skewed then we use median.
- Less than 5% data is missing (Appling condition).
#### Advantage:
- Easy To Implement.
#### Disadvantage:
- Don't preserve variable distribution.
- Add outliers.
- Change corelation `df.corr()` and covariance `df.cov()` with other columns.

## **3. Arbitrary Value Imputation:**

`এখানে, আমরা valid কিছু ভ্যালু দিয়ে NaN Value fill করি । যেমনঃ Age এর ক্ষেত্রে যেইখানে NaN ভ্যালু থাকবে আর পরবর্তীতে -১, ১০০০ ব্যবহার করতেই পারি । `

- If data is MAR.

#### Advantage:
- Easy To Implement.
#### Disadvantage:
- Don't preserve variable distribution.
- Change corelation `df.corr()` and covariance `df.cov()` with other columns.

## **4. End of Distribution Imputation:**

- We fill the NaN value with (mean-3*S.D) or (mean+3*S.D) if distribution is not skewed.
- If distribution is skewed, we use (Q1 - 1.5*IQR) or (Q1 + 1.5*IQR) we call it IQR Proximity.
- IQR (Q3 - Q1)
- Advantage and Disadvantage like 3(Arbitrary Value Imputation)

## **5. Random value Imputation:**

- We want to fill the NaN value from Age column, here we select random value from Age column.
- Applied for both numerical and categorical Feature.
- No inbulit function in sk-learn. We implement it through pandas.

#### Advantange:
- Distribution preserve.
#### Disadvantage:
- While deploying, we have to save the X_trian data in server.

## **6. Mode or Most-Frequent:**

- If data is MCAR(Missing Completly at Random)
- Replace NaN value with Most-Frequent Value.
- Inbuild function in sk-learn.

#### Advantage:
- Easy To Implement.

## **7. Missing Indicator:**

- In Country Column We have missing Value.
- We will create another column Name `Country_Missing`.
- `Country_Missing` we fill the column with Missing where we found NaN.

#### Advantage:
- Sometimes it improve the performance of the model.


## **8. KNN Imputer:**

- It is a multivariate analysis.
- Based on KNN(K-Nearest-Neighbour) algorithrm based.
- We calculate Nan_Euclidean_Distance .
- We can select the number of neighbor. (Most similar row, 1, 2 or something).
- Two types of KNN Imputer. Distance base and Weight base.
- If we have 5 columns, we fill the missing value by the help of 4 others columns.

#### Advantage:
- We get more accurate value.
#### Disadvantage:
- **Slow:** This process is slow. Each time we calculate Nan_Euclidean_Distance.
- **Memory High:** While deploying, we store the X_train value in server.

## **9. Iterative Imputer:**

- Base on MISE(Multivariate Imputation by Chained Equation) algorithrm.
- Perform better when data is MAR.

#### Advantage:
- We get more accurate value.
#### Disadvantage:
- **Slow:** This process is slow. Each time we calculate Nan_Euclidean_Distance.
- **Memory High:** While deploying, we store the X_train value in server.

**Summary:**
` Missing value impute করার পর আমরা মোটামুটি ৪ টা জিনিস দেখবো  i) corelation ii) covariance iii) Distribution iv) outliers.`


---
---

<br>

# Step-3: Feature Scaling:

## **Type of Feature Scaling:**
- Standardization 
- Normalization
    - MinMax Scaling
    - Mean Normalization
    - MaxAbs Scaling
    - Robust Scaling

**NOTE:**` কিছু কিছু algorithrm আছে যারা distance calculate করে same কলামের 2টা datapoint এর মাঝে । যদি দুইটা পয়েন্ট pt1(12,14) এবং pt2(120000,13) হয় তাহলে, distance = rootover((X1-X2)^2 + (Y1-Y2)^2) এখানে (12-10000)^2  calculate করার সময় X2 এর value অনেক বড় হওয়ার X2 dominate করবে । যেহেতু, ml model সংখ্যা ছাড়া তেমন কিছু বুঝে নাই তাই X2 কে প্রায়োরিটি বেশি দিবে। i) K-Mean ii) KNN iii) PCA(Principal Component Analysis) iv) ANN (Artificial Neural Network) v) Gradient Descent এই algorithrm গুলোর জন্য অবশ্যই অবশ্যই feature scaling করতে হবে । `

## **1. Standardization:**

- Also called Z-Score Normalization.
- formula, $\bar{X_i}$ = $\frac{X_i - \bar{X}} {\sigma}$
- where, $\bar{X_i}$ scaled value.
- $X_i$ current observation.
- $\bar{X}$ mean of the dataset.
- $\sigma$ Standard Deviation.
- In, sklearn we have `StandardScaler()`
- After applying Standardization our Standard Deviation,($\sigma$) = 1 and mean ($\bar{X}$) = 0

## **2. MinMax Scaling:**

- formula, $\bar{X_i}$ = $\frac{X_i - X_{min}} {X_{max} - X_{min}}$
- $X_i$ current observation.
- $X_{min}$ minimum value of the dataset.
- $X_{max}$ maximum value of the dataset.
- range between(0,1).
- In, sklearn we have `MinMaxScaler()`

## **3. Mean Normalization:**

- formula, $\bar{X_i}$ = $\frac{X_i - X_{mean}} {X_{max} - X_{min}}$
- $X_i$ current observation.
- $X_{mean}$ mean value of the dataset.
- $X_{min}$ minimum value of the dataset.
- $X_{max}$ maximum value of the dataset.
- No inbulid function in `sk-learn`.
- Instead of Mean Normalization we use Standardization.

## **4. MaxAbs Scaling:**

- formula, $\bar{X_i}$ = $\frac{X_i} {|X_{max}|}$
- When we have sparse data then we use MaxAbs Scaling.
- In, sklearn we have `MaxAbsScaler()`

## **5. Robust Scaling:**

- formula, $\bar{X_i}$ = $\frac{X_i - X_{median}} {\text{IQR}}$
- IQR(Inter Quartile Range) -> (${75}^{th} percentile - {25}^{th} percentile$)
- If we have outliers in our data then we use `Robust Scaling.`

## **Standardization vs Normalization:**

`আমরা maximum time ঐ Standardization ব্যবহার করি । যদি আমাদের কাছে minimum and maximum value জানা থাকে যেমনঃ image এর ক্ষেত্রে value (0~255) এই range হয়ে থাকে । এই ক্ষেত্রে আমরা Normalization ব্যবহার করি ।  CNN এ প্রচুর Normalization ব্যবহার করা হয় । `

---
---

<br>

# Step-4: Improvement:

## **1) Applying Mathematical Transformation:**

**NOTE:**
`ML Algorithrm Like: Regression, Linear Regression, Logistic Regression এ Normal Distribution Data দিলে ভালো আসে । এইক্ষেত্রে আমরা Mathematical Transformation, কোন কলামের উপর mathematical formula apply করে transformation করবো । আমরা এইগুলো Function transformation, Power transformation এবং Quantile transformation এর under এ পড়বো । `

### Types of Mathematical Transformation:

- Function Transformation
    - Log Transformation
    - Reciprocal
    - Square 
    - Square Root 
- Power Transformation 
    - Box-Cox
    - Yeo-Johnson
- Quantile Transformation


<br>

# <-----------More Improvement from deepseek-Ai----------------->

<br>

### **Suggestions from deepseek:**
1. **অটোমেশন টুলস:**
   - আপনি Pandas Profiling বা Sweetviz এর মতো অটোমেশন টুলস ব্যবহার করতে পারেন। এটি EDA প্রক্রিয়াটিকে আরও দ্রুত এবং সহজ করে তুলবে।

2. **ডেটা ভিজুয়ালাইজেশন:**
   - আপনি আরও এডভান্সড ভিজুয়ালাইজেশন টুলস (যেমন Plotly, Seaborn) ব্যবহার করতে পারেন। এটি ডেটার মধ্যে লুকানো প্যাটার্ন এবং সম্পর্কগুলি আরও ভালোভাবে বুঝতে সাহায্য করবে।

3. **ফিচার ইঞ্জিনিয়ারিং:**
   - আপনি ফিচার ইঞ্জিনিয়ারিং এর উপর আরও ফোকাস করতে পারেন। যেমন, নতুন ফিচার তৈরি করা, ফিচার সিলেকশন, এবং ফিচার ইন্টারঅ্যাকশন। এটি মডেলের পারফরম্যান্স আরও উন্নত করতে সাহায্য করবে।

4. **ডকুমেন্টেশন:**
   - আপনার EDA প্রক্রিয়াটি ভালোভাবে ডকুমেন্ট করুন। এটি আপনার টিম বা ক্লায়েন্টের সাথে শেয়ার করার সময় খুবই সাহায্য করবে।


### **More to find the importance of columns:**

হ্যাঁ, আপনি Random Forest ব্যবহার করে feature importance (কলামের ওজন) বের করতে পারেন এবং এটি EDA (Exploratory Data Analysis) এর একটি সাধারণ পদ্ধতি। তবে কিছু বিষয় মাথায় রাখা জরুরি:

### 1. **Null Values এর সমস্যা:**
   - Random Forest সাধারণত null values (NaN) হ্যান্ডল করতে পারে না। আপনি যদি null values সহ ডেটাসেটে Random Forest প্রয়োগ করেন, তাহলে এটি error throw করতে পারে।
   - সমাধান হিসেবে, null values আগে থেকে হ্যান্ডল করুন। যেমন:
     - Null values গুলো fill করতে পারেন (mean, median, mode, বা অন্য কোনো পদ্ধতি ব্যবহার করে)।
     - অথবা null values যুক্ত row/column ড্রপ করতে পারেন (যদি সেটি যুক্তিসঙ্গত হয়)।

### 2. **Feature Importance এর সীমাবদ্ধতা:**
   - Random Forest এর feature importance কলামের ওজন নির্দেশ করে, কিন্তু এটি সবসময় সম্পূর্ণ সঠিক বা ব্যাখ্যাযোগ্য নয়।
   - যদি কলামগুলোর মধ্যে উচ্চ correlation থাকে, তাহলে feature importance বিভ্রান্তিকর হতে পারে।
   - Feature importance শুধুমাত্র model-based importance দেখায়, এটি ডেটার প্রকৃত গঠন বা সম্পর্ককে পুরোপুরি প্রতিফলিত নাও করতে পারে।

### 3. **EDA এর অন্যান্য পদ্ধতি:**
   - Random Forest ছাড়াও অন্যান্য পদ্ধতি ব্যবহার করে কলামের গুরুত্ব বের করতে পারেন। যেমন:
     - Correlation matrix (numerical ডেটার জন্য)।
     - Chi-square test (categorical ডেটার জন্য)।
     - Mutual information।
     - অন্যান্য feature selection পদ্ধতি (L1 regularization, PCA ইত্যাদি)।

### 4. **Overfitting এর ঝুঁকি:**
   - Random Forest প্রয়োগ করার সময় overfitting এর ঝুঁকি থাকতে পারে, বিশেষ করে যদি ডেটাসেট ছোট হয় বা কলামের সংখ্যা বেশি হয়।
   - Cross-validation বা train-test split ব্যবহার করে model evaluation করুন।

### 5. **Scaling এর প্রয়োজন নেই:**
   - Random Forest এর জন্য সাধারণত ডেটা স্কেলিং (normalization/standardization) এর প্রয়োজন হয় না, কারণ এটি decision tree-based মডেল।



