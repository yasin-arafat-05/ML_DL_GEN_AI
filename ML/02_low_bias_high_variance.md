<br>

# `# Here i list down all the algo which have LOW-BIAS but High_Variance.`

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


