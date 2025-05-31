<br>

# `#01  LOW-BIAS but High_Variance Algorithrm: `

<br>

In machine learning, algorithms with **low bias and high variance** are typically those that are highly flexible and can fit complex patterns in the data, but they tend to overfit, meaning they perform well on training data but poorly on unseen test data due to sensitivity to noise or small changes in the dataset. These are often **non-linear** and **complex models**. Here’s a list of such algorithms:

### Algorithms with Low Bias and High Variance
1. **Decision Trees (Unpruned)**  
   - A single decision tree, especially when grown deep without pruning, can model intricate patterns in the data (low bias) but is prone to overfitting (high variance).

2. **k-Nearest Neighbors (k-NN) with Small k**  
   - When k is small (e.g., k=1), k-NN makes predictions based on very local data points, leading to low bias but high variance as it’s sensitive to noise or outliers.

3. **Support Vector Machines (SVM) with High Complexity (e.g., RBF Kernel with Small Gamma)**  
   - SVMs with a radial basis function (RBF) kernel and a small gamma value can fit the training data very closely (low bias) but may overfit, resulting in high variance.

4. **Neural Networks (Deep Learning Models)**  
   - Deep neural networks, especially with many layers and parameters (e.g., CNNs, RNNs), have low bias due to their expressive power but high variance unless regularized (e.g., dropout, weight decay).

5. **Random Forest (with Shallow Trees or Overfitting Settings)**  
   - While Random Forests typically reduce variance via averaging, a Random Forest with very few trees or overly deep trees can still exhibit higher variance than desired, though it’s less extreme than a single decision tree.

6. **Gradient Boosting (Unregularized or Overfit)**  
   - Without regularization (e.g., no limit on tree depth, high learning rate), gradient boosting models like XGBoost or LightGBM can fit the training data too closely (low bias) and overfit (high variance).

7. **Polynomial Regression (High Degree)**  
   - Polynomial regression with a high-degree polynomial (e.g., degree 10) can capture complex relationships (low bias) but often overfits, leading to high variance.

8. **Overparameterized Linear Regression (e.g., with Many Features)**  
   - If the number of features is very large compared to the dataset size (e.g., in high-dimensional settings), linear regression can become overly flexible, reducing bias but increasing variance.

### Key Characteristics
- **Low Bias**: These models are flexible and can fit a wide range of functions or patterns in the data.
- **High Variance**: They are sensitive to small changes in the training data, leading to overfitting without proper regularization or constraints.

### Mitigating High Variance
To address the high variance in these models, techniques like regularization (e.g., L1/L2 penalties), pruning, increasing k in k-NN, reducing model complexity, or using ensemble methods (e.g., bagging or boosting with care) can be applied.

`**আচ্ছা low bias and high variance model গুলোর তে low bias low variance এ আনার জন্য কোন কোন ensemble learning এর  bagging and boosting ব্যবহার  করি ।**`

<br>

# `#02 Effect of outliers:`

<br>

- Linear Regression, Logistic regresion, Adaboost,Deep Learning এ outliers থাকলে model ভালো performance করে না । 

- Tree based Algorithrm like: decision tree,random forest,XGBoost এ outliers এর কোন effect থাকে না । 



<br>

# `#03 Effect of scaling:`

<br>

`1. **Don't Need Scaling**:`

কিছু মডেল আছে, যেগুলো **feature scaling** বা **data normalization** এর ওপর কম প্রভাবিত হয়। সেগুলো হল:

- **Tree-based algorithms**:
  - **Decision Trees**
  - **Random Forest**
  - **XGBoost**
  - **LightGBM**
  - **CatBoost**

  এই মডেলগুলোর উপর **feature scaling** বা **normalization** এর তেমন কোনো প্রভাব পড়ে না, কারণ তারা ডেটার মধ্যে **split** করে, অর্থাৎ তারা মূলত ডেটার **relative ordering** বা **values** দেখে কাজ করে।

- **Rule-based models**: 
  - **K-Nearest Neighbors (KNN)**:  কিছুটা প্রভাব ফেলতে পারে, তবে সাধারণত **tree-based models** এর মতোই কাজ করে, যা **scaling** এর খুব বেশি প্রভাব নাও পেতে পারে।

<br>

`2. **Need Scaling**:`

যেসব মডেল **feature scaling** এর ওপর বেশি নির্ভরশীল:
- **Linear Regression**
- **Logistic Regression**
- **SVM (Support Vector Machines)**
- **KNN (K-Nearest Neighbors)**
- **Neural Networks (Deep Learning)**

এই মডেলগুলির জন্য **feature scaling** খুবই গুরুত্বপূর্ণ কারণ তারা ডেটার মধ্যে **distance** বা **dot product** ব্যবহার করে কাজ করে, এবং যখন বিভিন্ন ফিচারের স্কেল ভিন্ন হয়, তখন মডেলটি ভুলভাবে প্রভাবিত হতে পারে।

<br>

