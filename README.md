# ML_MNL
ML_MNL_model_comparison
## Transformer
In the folder **Transformer**, there are mainly two python documents and one folder. They are used to train transformer and contain the final trained model named simulator.pth.
1. **transformer.py** includes the whole class of modified transformer of our paper.
2. **train_transformer.py** contains the process of training and testing our modified transformer with specific hyperparameter.
3. Folder called **simulator** holding the final trained model "**simulator.pth**" .
## Choice Models
In the folder **Choice_Models_Training_and_Comparison**, there are mainly four python documents and one jupyter notebook documents. They are used to upload the files used to train choice models, generate new training dataset, test dataset and decision-test product dataset with real data and compare different choice models' prediction power and decision power.
1. **assortment_process.py** provides the method to get the assortment decisions of different choice models.
2. **build_new_data.py** holds all methods used to generate new training dataset, test dataset and decision-test product dataset with real data.
3. **choicemodels.py** contains all classes of different choice models such as DeepFM, DeepFM-a and MNL. 
4. **training_function.py** contains the process of training and testing choice models (DeepFM, DeepFM-a and MNL). And for MNL, we trained MNL with specific learning rate.
5. **experiment_process.ipynb** contains the whole process described in our paper from **generating data**, **training and testing choice models**, and finally **get assortment decision of those choice models**.
