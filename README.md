# ML_MNL
In our paper, we used two different dataset to perform a validation on our framework. One dataset is actual air ticket reservation data collected by Amadeus \([Mottini & Acuna-Agost 2017](https://dl.acm.org/doi/abs/10.1145/3097983.3098005)\). The other dataset is a published real hotel dataset provided
by Expedia which are used in [Personalize Expedia Hotel Searches -
ICDM 2013 Kaggle](https://www.kaggle.com/competitions/expedia-personalized-sort/overview). 

## Airline
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
### Expedia_preprocess
In the folder **Expedia_preprocess**, we perform a preprocess function on original data downloaded in Kaggle, and finally get the document **df_OR_normal_process_with_nonpurchase_clean.csv** as our dataset. 

### Expedia_Transformer
In the folder **Expedia_Transformer**, there are mainly eight python documents and two folder. They are used to show the whole structure of our transformer and contain the final trained model.
1. **ChoiceTfData.py** includes the structure of data which are used in our Transformer.
2. **ChoiceTransformer.py** contains the structure of our Transformer which are used to predict the chosen probability for products in one assortment.
3. **PurchaseTransformer.py** contains the structure of our Transformer which are used to predict the chosen probability for assortments.
4. **Transformer.py** includes the whole class of modified transformer used in ChoiceTransformer.
5. **Transformer_nonpurchase.py** includes the whole class of modified transformer used in PurchaseTransformer.
6. **true_model.py** includes the whole process 
3. Folder called **simulator** holding the final trained model "**simulator.pth**" .
### Choice Models
In the folder **Expedia_Choice_Models_Training_and_Comparison**, there are mainly four python documents and one jupyter notebook documents. They are used to upload the files used to train choice models, generate new training dataset, test dataset and decision-test product dataset with real data and compare different choice models' prediction power and decision power.
1. **assortment_process.py** provides the method to get the assortment decisions of different choice models.
2. **build_new_data.py** holds all methods used to generate new training dataset, test dataset and decision-test product dataset with real data.
3. **choice model comparison with real data** use real data to compare models' prediction power. 
4. **choicemodels.py** contains all classes of different choice models such as DeepFM, DeepFM-a and MNL. 
5. **choicemodels_real_data.py** contains all classes of different choice models such as DeepFM, DeepFM-a, MNL, MMNL and Exponomial Choice Model used in real data comparison.
6. **experiment_process.ipynb** contains the whole process described in our paper from **generating data**, **training and testing choice models**, and finally **get assortment decision of those choice models**.
7. **training_function.py** contains the process of training and testing choice models (DeepFM, DeepFM-a and MNL). And for MNL, we trained MNL with specific learning rate.
8. **training_function_real_data.py** contains the process of training and testing choice models (DeepFM, DeepFM-a, MNL, MMNL and Exponomial Choice Model).