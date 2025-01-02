import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.impute import KNNImputer
import streamlit as st

# Streamlit UI for file upload
st.title("Task Allocation System")

# File uploader

uploaded_file = st.file_uploader("Upload your data (CSV file)", type=["csv"])


if uploaded_file is not None:
    # Load and preprocess the uploaded data
    data = pd.read_csv(uploaded_file)
    data['Skills'] = data['Skills'].fillna('')
    data['Interests'] = data['Interests'].fillna('')
    data['Experience'] = data['Experience'].fillna('Unknown')
    data = data.drop_duplicates()
    # Encode skills and interests
    mlb_skills = MultiLabelBinarizer()
    mlb_interests = MultiLabelBinarizer()

    # Splitting the skills and interests into lists
    data['Skills'] = data['Skills'].apply(lambda x: x.split(', ') if x != '' else [])
    data['Interests'] = data['Interests'].apply(lambda x: x.split(', ') if x != '' else [])

    # Apply binarization on skills and interests
    skills_encoded = mlb_skills.fit_transform(data['Skills'])
    interests_encoded = mlb_interests.fit_transform(data['Interests'])

    skills_df = pd.DataFrame(skills_encoded, columns=mlb_skills.classes_)
    interests_df = pd.DataFrame(interests_encoded, columns=mlb_interests.classes_)

    skills_df = pd.DataFrame(skills_encoded, columns=[f"Skill_{col}" for col in mlb_skills.classes_])
    interests_df = pd.DataFrame(interests_encoded, columns=[f"Interest_{col}" for col in mlb_interests.classes_])

    # Valid skills and interests
    valid_skills = set(mlb_skills.classes_)
    valid_interests = set(mlb_interests.classes_)

    # Valid skills and interests
    valid_skillslist = mlb_skills.classes_
    valid_interestslist = mlb_interests.classes_
    flattened_skills = list(set(skill.strip() for skills in valid_skillslist for skill in skills.split(',')))
    flattened_skills.sort()  # Sort the list alphabetically for easier reading
    #print("Unique Flattened Valid Skills:", flattened_skills)

    # Flatten and remove duplicates from interests
    flattened_interests = list(set(interest.strip() for interests in valid_interestslist for interest in interests.split(',')))
    flattened_interests.sort()  # Sort the list alphabetically for easier reading
    #print("Unique Flattened Valid Interests:", flattened_interests)

    # Encode experience
    le_experience = LabelEncoder()
    data['Experience'] = le_experience.fit_transform(data['Experience'])
    # Combine datasets
    data_combined = pd.concat([data, skills_df, interests_df], axis=1)
    data_combined = data_combined.fillna(0)
    data_combined.head()

    #print(data_combined.columns[data_combined.columns.duplicated()])
    # Scale and impute features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data_combined.drop(columns=['Name', 'Team', 'Skills', 'Interests', 'Past Contribution']))

    imputer = KNNImputer(n_neighbors=5)
    scaled_features = imputer.fit_transform(scaled_features)

    kmeans = KMeans(n_clusters=15, random_state=42)
    data_combined['Cluster'] = kmeans.fit_predict(scaled_features)

    # Apply PCA to reduce the data to 2 dimensions
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_features)

    # Create a DataFrame to store PCA results
    pca_df = pd.DataFrame(principal_components, columns=['PC1', 'PC2'])
    data_combined['Cluster'] = kmeans.fit_predict(principal_components)
    pca_df['Cluster'] = data_combined['Cluster']  # Assuming original cluster labels in 'data_combined'

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='viridis', s=100, edgecolor='black')
    plt.title('KMeans Clusters with PCA Projection', fontsize=16)
    plt.xlabel('Principal Component 1', fontsize=12)
    plt.ylabel('Principal Component 2', fontsize=12)
    plt.legend(title='Cluster')
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.countplot(data=data_combined, x='Experience', hue='Cluster', palette='Set2')
    plt.title('Distribution of Experience Across Clusters', fontsize=16)
    plt.xlabel('Experience (Encoded)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.show()

    # Step 1: Apply KMeans clustering on PCA-transformed data with 3 clusters (based on your choice)
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(principal_components)


    # Step 2: Calculate silhouette score for this clustering
    silhouette_avg = silhouette_score(principal_components, kmeans.labels_)
    st.write(f"Silhouette Score after PCA: {silhouette_avg:.2f}")
    
    # Step 3: Assign cluster labels to data
    pca_df['Predicted_Cluster'] = kmeans.labels_

    # Visualize the clusters after clustering
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='PC1', y='PC2', hue='Predicted_Cluster', data=pca_df, palette='viridis', s=100, edgecolor='black')
    plt.title('KMeans Clusters after Clustering (PCA)', fontsize=16)
    plt.xlabel('Principal Component 1', fontsize=12)
    plt.ylabel('Principal Component 2', fontsize=12)
    plt.legend(title='Predicted Cluster')
    plt.show()

    # Step 4: Check clustering performance (using silhouette scores for different k)
    silhouette_scores = []
    for k in range(2, 11):
        kmeans_temp = KMeans(n_clusters=k, random_state=42)
        kmeans_temp.fit(principal_components)
        silhouette_scores.append(silhouette_score(principal_components, kmeans_temp.labels_))
    print(f"Silhouette Scores for Different Clusters (PCA): {silhouette_scores}")

    plt.figure(figsize=(8, 6))
    plt.plot(range(2, 11), silhouette_scores, marker='o')
    plt.title('Silhouette Scores for Different Clusters (PCA-transformed data)')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.show()

    inertia = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(principal_components)
        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, 11), inertia, marker='o')
    plt.title('Elbow Method for Optimal Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.show()

    # Step 5: Output the predicted clusters
    data_combined['Predicted_Cluster'] = kmeans.labels_

    from sklearn.metrics.pairwise import cosine_similarity

    def allocate_task(event_name, required_skills, required_interests, required_member_count):
        # Filter required skills and interests
        required_skills = [skill for skill in required_skills if skill in valid_skills]
        required_interests = [interest for interest in required_interests if interest in valid_interests]

        if not required_skills and not required_interests:
            print("No matching skills or interests found in the data.")
            return None, 0

        # Create vectors for required skills and interests
        required_skills_vector = np.zeros(len(mlb_skills.classes_))
        required_interests_vector = np.zeros(len(mlb_interests.classes_))

        for skill in required_skills:
            required_skills_vector[mlb_skills.classes_.tolist().index(skill)] = 1

        for interest in required_interests:
            required_interests_vector[mlb_interests.classes_.tolist().index(interest)] = 1

        # Combine skills and interests vectors, giving more weight to skills
        required_features_vector = np.concatenate([required_skills_vector * 2, required_interests_vector])

        # Calculate cosine similarity between required features and member features
        member_features = data_combined[[f"Skill_{col}" for col in mlb_skills.classes_] + [f"Interest_{col}" for col in mlb_interests.classes_]].values
        similarities = cosine_similarity([required_features_vector], member_features)

        # Get indices of members sorted by similarity
        sorted_indices = similarities.argsort()[0][::-1]

        # Filter members with all required skills
        filtered_indices = [i for i in sorted_indices if all(data_combined.iloc[i][f"Skill_{skill}"] == 1 for skill in required_skills)]

        # Select suitable members based on filtered indices and required count
        if filtered_indices:
            suitable_members = data_combined.iloc[filtered_indices][:required_member_count]
        else:
            # If no perfect match, select from sorted indices (next best match)
            suitable_members = data_combined.iloc[sorted_indices][:required_member_count]
            print("Warning: No perfect skill match found. Selecting next best suitable members.")

        return suitable_members[['Name', 'Skills', 'Interests']], len(suitable_members)

    event_name = st.text_input("Enter some text:")
    required_skills = st.multiselect("Select required skills", options=flattened_skills)
    required_interests = st.multiselect("Select required interests", options=flattened_interests)
    required_member_count = st.number_input("Enter number of required members", min_value=1, step=1)

    if st.button("Allocate Task"):
            suitable_members, suitable_members_count = allocate_task(
                event_name,required_skills, required_interests, required_member_count
            )
            
            if suitable_members is not None:
                st.write("Suitable members for the task:",event_name)
                st.write(suitable_members)
                st.write(f"Total available members: {suitable_members_count}")
            else:
                st.write("No suitable members found.")
   
