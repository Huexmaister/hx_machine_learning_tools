from setuptools import setup, find_packages

setup(
    name="hx_machine_learning_tools",
    version="2025_09_25",
    author="Daniel Sarabia aka Huexmaister",
    author_email="dsarabiatorres@gmail.com",
    description="Librería de utilidades de modelos de machine learning",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Huexmaister/hx_machine_learning_tools",
    packages=find_packages(),  # Busca automáticamente todos los paquetes
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "catboost==1.2.8",
        "category_encoders==2.8.1",
        "constants_and_tools==1.1.2",
        "imbalanced-learn==0.14.0",
        "imblearn==0.0",
        "joblib==1.5.2",
        "scikit-learn==1.7.2",
        "shap==0.48.0",
        "xgboost==3.0.5",
        "lightgbm==4.6.0",
    ],
    python_requires=">=3.10",
)
