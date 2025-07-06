## Aerodynamic Optimization Using Neural Network

This project uses a deep neural network trained on high-fidelity **CFD (Computational Fluid Dynamics)** simulations to predict:

* Coefficient of Lift (CL)
* Coefficient of Drag (CD)
* Optimal aerodynamic parameters to maximize the lift-to-drag ratio (CL/CD)

The goal is to enable fast, accurate aerodynamic analysis and optimization without the computational cost of running CFD for each new configuration.

**Neural Network Architecture**
![image](https://github.com/user-attachments/assets/fb3a9745-c758-4eb3-8544-e23f35985515)


---

## Background

The model has been trained using actual CFD simulation datasets, specifically focused on the **DAE11 airfoil**. This airfoil is commonly used in UAV and high-performance aircraft designs, making it a practical target for data-driven aerodynamic modeling.

**DAE11 airfoil**
![Screenshot 2025-07-06 204829](https://github.com/user-attachments/assets/afaa93fc-ea4f-46cf-8fc6-5c4aa8a52c20)

---

## Features

* Predicts CL and CD for any given combination of angle of attack and Mach number
* Uses a fully connected neural network with two hidden layers (64 ReLU units each)
* Provides optimal parameter combinations based on CL/CD ratio
* Allows exporting simulation predictions to CSV
* Supports K-Fold cross-validation for model reliability

---

## Model Architecture

```
Input: [Alpha, Mach]
↓
Dense(64, relu)
↓
Dense(64, relu)
↓
Output: [CL, CD, Cp]
```

* Loss Function: Mean Squared Error (MSE)
* Optimizer: Adam
* Evaluation Metric: Mean Absolute Error (MAE)

---

## Files and Functions

| File / Function                       | Description                                                                                       |
| ------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `get_best_params(angles, velocities)` | Returns a DataFrame of CL, CD, and CL/CD ratio for input combinations, sorted by best performance |
| `cfd_simulated_dataset.csv`    | Generated dataset from running multiple CFD Simulations.                                                   |
| `model`                               | Trained Keras/TensorFlow model for aerodynamic prediction                                         |
| `KFold` code block                    | Runs 5-fold cross-validation to assess generalization                                             |

---

## Example Usage

### Get Best Aerodynamic Configurations

```python
angles = [0, 5, 10, 15, 20]
velocities = [0.2, 0.3, 0.4, 0.5]

results = get_best_params(angles, velocities)
print(results.head())  # Show best-performing configs
```



---

## Model Evaluation

Performed using 5-fold cross-validation:

```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
```
Each fold trains and validates the model, measuring **MAE** on unseen data to ensure robust generalization.


**Prediction vs Actual Values**

![Screenshot 2025-07-04 214318](https://github.com/user-attachments/assets/bc4c77f3-5470-4c0b-ad62-55572ab9445f)

![Screenshot 2025-07-04 214452](https://github.com/user-attachments/assets/b0d422b6-b050-4549-9952-495ecb84ecd5)

---

## Sample Output

| Alpha | Mach | CL   | CD    | CL/CD |
| ----- | ---- | ---- | ----- | ----- |
| 10.0  | 0.3  | 1.25 | 0.045 | 27.78 |
| 15.0  | 0.4  | 1.38 | 0.065 | 21.23 |

---

## Requirements

* Python 3.7+
* TensorFlow / Keras
* NumPy, Pandas
* (Optional) `google.colab` for notebook-based workflows

---

## Future Work

I plan to continue expanding and improving this project with the following goals:

1. Add support for multiple airfoils beyond DAE11
2. Improve model accuracy through hyperparameter tuning and deeper architectures
3. Include additional aerodynamic parameters such as **pressure coefficient (Cp)** and moment coefficients

---

## Motivation

Running CFD simulations for every new aerodynamic condition is computationally expensive. This project allows me to learn from those simulations and make rapid predictions, enabling faster design and optimization loops in aerospace applications.

---

## License

This project is intended for research and academic use. Please cite appropriately if used in any publications or derivative work.

