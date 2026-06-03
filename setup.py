from setuptools import setup, find_packages

setup(
    name="ridesense_package",
    version="0.0.1",
    author="Raj Kiran Reddy",
    description="RideSense AI — Ride Fare Prediction ML Pipeline",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "streamlit",
    ]
)