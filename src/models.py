"""
Classification model builders and pipelines.
Provides Logistic Regression (with feature scaling) and Decision Tree models.
"""

from typing import Dict, Any
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def build_logistic_regression(
    C: float = 1.0, 
    max_iter: int = 1000, 
    random_state: int = 42
) -> Pipeline:
    """
    Build a Logistic Regression pipeline with StandardScaler.
    
    StandardScaler ensures that features on large numerical scales do not
    dominate the L2/L1 penalty regularization or gradient steps.
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            C=C, 
            max_iter=max_iter, 
            random_state=random_state, 
            solver='lbfgs'
        ))
    ])


def build_decision_tree(
    max_depth: int = 4, 
    min_samples_split: int = 5,
    min_samples_leaf: int = 2,
    criterion: str = 'gini',
    random_state: int = 42
) -> DecisionTreeClassifier:
    """
    Build a Decision Tree Classifier with depth constraints to prevent overfitting.
    """
    return DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,
        random_state=random_state
    )


def train_all_models(X_train, y_train, random_state: int = 42) -> Dict[str, Any]:
    """
    Train both Logistic Regression and Decision Tree models on the training set.
    
    Returns:
        Dictionary mapping model names to fitted model objects.
    """
    models = {
        "Logistic Regression": build_logistic_regression(random_state=random_state),
        "Decision Tree": build_decision_tree(random_state=random_state)
    }
    
    fitted_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted_models[name] = model
        
    return fitted_models
