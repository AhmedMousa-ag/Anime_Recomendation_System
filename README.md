# Anime_Recomendation_System_GCP
Implementing NNFC recomendation system on GCP with periodic training.
I did slight change with Neural Network to predict rating istead of predicting if the user will like it or not. "You can change it back to match the paper by uncomment suggested lines in Utils/data/preprocess.py and train/colabritive_system.py".

This project is due to participating in DataTalks.club project of the week. 

# Navigating Code:

You can test the code on your device without cloud by opening testing_code.py and on cloud with explaination GCP_kubeflow_depl.ipynb.

Utils folder is where to find the preprocess modules to preprocess the data.

Train folder where to find the recommendation system code.

![image](https://user-images.githubusercontent.com/59775002/200170513-92c772e5-df69-4741-b0ea-5ee50a98163d.png)


# Dataset can be found on Kaggle:
https://www.kaggle.com/datasets/hernan4444/anime-recommendation-database-2020
