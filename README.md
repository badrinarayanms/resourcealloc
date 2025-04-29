# Resource Allocation System

## Overview
A Streamlit-based web application that intelligently allocates tasks to team members based on their skills, interests, and experience using machine learning clustering techniques.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data Format](#data-format)
- [Technical Approach](#technical-approach)
- [Dependencies](#dependencies)
- [Screenshots](#screenshots)
- [Future Enhancements](#future-enhancements)
- [License](#license)

## Features

### Data Processing
- Handles missing values in skills, interests, and experience fields
- Multi-label encoding for skills and interests
- Experience level encoding using LabelEncoder
- Feature scaling with StandardScaler
- Missing value imputation using KNNImputer

### Machine Learning Components
- KMeans clustering for grouping similar team members
- PCA for dimensionality reduction and visualization
- Silhouette scoring for cluster evaluation
- Elbow method for optimal cluster determination
- Cosine similarity for task-member matching

### Task Allocation
- Dynamic matching of required skills and interests
- Configurable number of required members
- Fallback to next-best matches when perfect matches aren't available
- Visual cluster analysis of team members

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/badrinarayanms/resourcealloc.git
   cd resourcealloc

2. Create and activate a virtual environment (recommended):
   ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. Install the required packages:    
    ```bash
    pip install -r requirements.txt


## Usage

### Running the Application
To launch the application, execute the following command in your terminal:

    ```bash
    streamlit run app.py

##Required CSV Structure
```bash
Name,Team,Skills,Interests,Experience,Past Contribution
