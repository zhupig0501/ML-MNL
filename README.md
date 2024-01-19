# Transformer-Choice-Model
In [our research paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4298996), we used two different dataset to perform a validation on our framework. One dataset is actual air ticket reservation data collected by Amadeus \([Mottini & Acuna-Agost 2017](https://dl.acm.org/doi/abs/10.1145/3097983.3098005)\). The other dataset is a published real hotel dataset provided
by Expedia which are used in [Personalize Expedia Hotel Searches -
ICDM 2013 Kaggle](https://www.kaggle.com/competitions/expedia-personalized-sort/overview). 

## Airline
### # Airline Synthetic Dataset Description
This repository hosts a synthetic dataset modeled on airline transactions. The data is not real but has been generated based on Algorithm 2 described in [our research paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4298996). The dataset includes two main files:
1. **transaction_data.csv**: Contains synthetic labeled transaction data for 30,000 assortments, with each assortment comprising between 5 to 30 products.
2. **pool_product.csv**: Provides 1000 synthetic product pools. Each pool contains 100 different products that can be used as candidates for an assortment.

This synthetic dataset is designed for analytical and modeling purposes in the context of airline transactions and customer behavior studies.

### Transformer
In the folder **Transformer**, there are mainly two python documents and one folder. They are used to train transformer and contain the final trained model named simulator.pth.
1. **transformer.py** includes the whole class of modified transformer of our paper.
2. **train_transformer.py** contains the process of training and testing our modified transformer with specific hyperparameter.
3. Folder called **simulator** holding the final trained model "**simulator.pth**" .
### Choice Models
In the folder **Choice_Models_Training_and_Comparison**, there are mainly six python documents and two jupyter notebook documents. They are used to upload the files used to train choice models, generate new training dataset, test dataset and decision-test product dataset with real data and compare different choice models' prediction power and decision power.
1. **assortment_process.py** provides the method to get the assortment decisions of different choice models.
2. **build_new_data.py** holds all methods used to generate new training dataset, test dataset and decision-test product dataset with real data.
3. **choice model comparison with real data.ipynb** use real data to compare models' prediction power. 
4. **choicemodels.py** contains all classes of different choice models such as DeepFM, DeepFM-a and MNL. 
5. **choicemodels_real_data.py** contains all classes of different choice models such as DeepFM, DeepFM-a, MNL, MMNL and Exponomial Choice Model used in real data comparison.
6. **experiment_process.ipynb** contains the whole process described in our paper from **generating data**, **training and testing choice models**, and finally **get assortment decision of those choice models**.
7. **training_function.py** contains the process of training and testing choice models (DeepFM, DeepFM-a and MNL). And for MNL, we trained MNL with specific learning rate.
8. **training_function_real_data.py** contains the process of training and testing choice models (DeepFM, DeepFM-a, MNL, MMNL and Exponomial Choice Model).


## Expedia
### Expedia Synthetic Hotel Dataset Description
This repository contains a synthetic dataset based on hotel transactions from Expedia. The data is not real but has been generated based on Algorithm 2 described in [our research paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4298996). The dataset is divided into two main files:
1. **transaction_data.csv**: This file includes synthetic labeled transaction data for 30,000 hotel room assortments. Each assortment comprises between 5 to 30 different products.
2. **pool_product.csv**: This file provides 1,000 synthetic room pools. Each pool offers a selection of 100 different products.

Tailored for the hospitality industry, this synthetic dataset is ideal for analyses and modeling, focusing on hotel transactions and customer choice behavior.
### Expedia_Transformer
In the folder **Expedia_Transformer**, there are mainly eight python documents and two folder. They are used to show the whole structure of our transformer and contain the final trained model.
1. **ChoiceTfData.py** includes the structure of data which are used in our Transformer, generated from the csv.
2. **ChoiceTransformer.py** contains the structure of our Transformer which are used to predict the chosen probability for products in one assortment.
3. **PurchaseTransformer.py** contains the structure of our Transformer which are used to predict the chosen probability for assortments.
4. **Transformer.py** includes the whole class of modified transformer used in ChoiceTransformer.
5. **Transformer_nonpurchase.py** includes the whole class of modified transformer used in PurchaseTransformer.
6. **true_model.py** contains integrated functions train_model and predict_model which can use csv to train model directly and use csv and model name to call specific model for prediction.
7. Folder called **checkpoints** holding the final trained models.
8. Folder called **data** holding the intermediate variables used for model training and predictions.

### Train_Transformer
**Train_Transformer.ipynb** demonstrates the procedure of utilizing Python and CSV files within the context of **Expedia_Transformer** for both model training and prediction.

### Choice Models
In the folder **Expedia_Choice_Models_Training_and_Comparison**, there are mainly six python documents and two jupyter notebook documents. They are used to upload the files used to train choice models, generate new training dataset, test dataset and decision-test product dataset with real data and compare different choice models' prediction power and decision power. The content in the folders are totally different from the folder in **Airline**, but have the same names.
1. **assortment_process.py** provides the method to get the assortment decisions of different choice models.
2. **build_new_data.py** holds all methods used to generate new training dataset, test dataset and decision-test product dataset with real data.
3. **choice model comparison with real data** use real data to compare models' prediction power. 
4. **choicemodels.py** contains all classes of different choice models such as DeepFM, DeepFM-a and MNL. 
5. **choicemodels_real_data.py** contains all classes of different choice models such as DeepFM, DeepFM-a, MNL, MMNL and Exponomial Choice Model used in real data comparison.
6. **experiment_process.ipynb** contains the whole process described in our paper from **generating data**, **training and testing choice models**, and finally **get assortment decision of those choice models**.
7. **training_function.py** contains the process of training and testing choice models (DeepFM, DeepFM-a and MNL). And for MNL, we trained MNL with specific learning rate.
8. **training_function_real_data.py** contains the process of training and testing choice models (DeepFM, DeepFM-a, MNL, MMNL and Exponomial Choice Model).
