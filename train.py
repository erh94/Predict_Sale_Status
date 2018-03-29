import sys
import matplotlib.pyplot as plt
import sklearn as sk
import seaborn as sns
import pandas as pd
import pickle as pk
import numpy as np
from sklearn.ensemble import RandomForestClassifier #
from sklearn.linear_model import SGDClassifier #
from sklearn.tree import DecisionTreeClassifier#
from sklearn.neighbors import KNeighborsClassifier#
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score #for_accuracy
from sklearn.multiclass import OneVsRestClassifier #multiclass classifier
from sklearn.externals import joblib


# In[2]:


def universalframe(columnlist):
    dummyframe = pd.DataFrame(columns=columnlist)
    MSZoning = ['A','C (all)','FV','I','RH','RL','RP','RM']
    Neighborhood = ['Blmngtn','Blueste','BrDale',
                'BrkSide','ClearCr','CollgCr',
                'Crawfor','Edwards','Gilbert','IDOTRR',
                'MeadowV','Mitchel','NAmes','NoRidge','NPkVill','NridgHt',
                'NWAmes','OldTown','SWISU','Sawyer','SawyerW',
                'Somerst','StoneBr','Timber','Veenker','Empty']
    Condition1 = ['Artery','Feedr','Norm','RRNn','RRAn','PosN','PosA','RRNe','RRAe']
    Condition2 = ['Artery','Feedr','Norm','RRNn','RRAn','PosN','PosA','RRNe','RRAe']
    BldgType = ['1Fam','2fmCon','Duplex','TwnhsE','Twnhs','TwnhsI']
    HouseStyle = ['1Story','1.5Fin','1.5Unf','2Story','2.5Fin','2.5Unf','SFoyer','SLvl']
    RoofStyle = ['Flat','Gable','Gambrel','Hip','Mansard','Shed']
    MasVnrType = ['BrkCmn','BrkFace','CBlock','None','Stone']
    GarageType = ['2Types','Attchd','Basment','BuiltIn','CarPort','Detchd','NA']
    GarageFinish =['Fin','RFn','Unf','NA']
    SaleStatus = ['SoldFast','SoldSlow','NotSold']
    SaleCondition = ['Normal','Abnorml','AdjLand','Alloca','Family','Partial']
    CategoricalColumns = ['MSZoning','Condition1','Condition2','BldgType','HouseStyle','RoofStyle'
                      ,'MasVnrType','GarageType','GarageFinish','SaleCondition','SaleStatus','Neighborhood']
    
    ListofList = [MSZoning,Condition1,Condition2,BldgType, HouseStyle,RoofStyle
                      ,MasVnrType,GarageType,GarageFinish,SaleCondition,SaleStatus,Neighborhood]


    dummyframe['Neighborhood']=pd.Series(Neighborhood)
    
    
    for column,columname in zip(ListofList,CategoricalColumns):
                      dummyframe[columname]=pd.Series(column)
    
        
    for i in columnlist:
        if i not in CategoricalColumns:
            dummyframe[i]=list(dummyframe.index)
    
    for column in dummyframe.columns:
        if(dummyframe[column].dtype == "object"):
            dummyframe[column].fillna(str('Empty'),inplace=True)
        elif(dummyframe[column].dtype!='object'):
                dummyframe[column].fillna(dummyframe[column][0],inplace=True)
    
        
    return dummyframe
    



def input_columns():
    return ['Id','MSSubClass','MSZoning','LotArea','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','OverallQual',
            'OverallCond','YearBuilt','YearRemodAdd','RoofStyle','MasVnrType','MasVnrArea','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','GarageType','GarageYrBlt','GarageFinish','GarageCars','GarageArea','SaleCondition',
            'SaleStatus'  ]

def pre_process(df):
    
    augment_df = universalframe(input_columns())
    
    df_original_num = len(df)   
    
    df = pd.concat(objs=[df, augment_df], axis=0)
    
    
    if('SaleStatus' in df.columns):
        df['SaleStatus'].fillna(df['SaleStatus'].mode(),inplace=True)
        df = df.drop('SaleStatus', axis=1)
    
    for column in df.columns:
        if(df[column].dtype == "object"):
                df[column].fillna(str('Empty'),inplace=True)
        elif (df[column].dtype!='object'):
                df[column].fillna(df[column].median(),inplace=True)
                
    
        
    
    one_hot_coded = pd.get_dummies(df)
    
    one_hot_coded = one_hot_coded[:df_original_num]
    
    columns = one_hot_coded.columns
    columns = [ x for x in columns if "Empty" not in x ]
    
    
    return one_hot_coded[columns]





def create_dataset(data,ratio,target):
        target = data['SaleStatus']
        pre_processed_features = pre_process(data)
        a=list(pre_processed_features.columns)
        a.remove('Id')
        features = pre_processed_features[a]
        min_max_scaler = sk.preprocessing.MinMaxScaler()
        np_scaled = min_max_scaler.fit_transform(features)
        features_normalized = pd.DataFrame(np_scaled)
        
        return train_test_split(features_normalized, target, test_size=ratio, random_state=42),features_normalized


# In[7]:

#this function performs the accuracy calculation and 5 fold cross validation 
def accuracy_calculation(classifier_object,x,y,X_train,y_train,X_test,y_test):
    scores = cross_val_score(classifier_object,x,y,cv=5)
    print("Cross-Validation Scores :",scores)
    print("Cross-Validation Accuracy :",scores.mean() , " Deviation(+/-) :",scores.std())
    svc_pred = classifier_object.predict(X_test)
    print("Accuracy of Test Data:",accuracy_score(y_test,svc_pred))
    svc_pred = classifier_object.predict(X_train)
    print("Accuracy on Train data :",accuracy_score(y_train,svc_pred))
    return


#this function start the analysis of classifier

def analysis(classifier_object,x,y,X_train,y_train,X_test,y_test,pkl):
    classifier_object.fit(X_train,y_train)
    accuracy_calculation(classifier_object,x,y,X_train,y_train,X_test,y_test)
    joblib.dump(classifier_object, pkl)
    return
    


def main():
    if len(sys.argv)< 2:
        print("Enter Proper Arguments")
        sys.exit(1)
    
    data = pd.read_csv(sys.argv[1])
    target = data['SaleStatus']
    data.drop('SaleStatus',axis=1,inplace=True)
    
    #preprocessing of data is performed here
    pre_processed_features = pre_process(data)
    a=list(pre_processed_features.columns)
    a.remove('Id')

    features = pre_processed_features[a]

    #normalization of data is performed here

    min_max_scaler = sk.preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(features)
    features_normalized = pd.DataFrame(np_scaled)
    X_train, X_test, y_train, y_test =train_test_split(features_normalized, target, test_size=0.2, random_state=42)
    
    # # SVC
    # print("\nAnalysis of SVC with hypertuned parameters")
    # pkl_filename = 'SVC.pkl'
    # SVC_classifier = SVC(kernel='rbf',C=300,decision_function_shape='ovo',gamma=0.1)
    # analysis(SVC_classifier,features_normalized,target,X_train,y_train,X_test,y_test,pkl_filename)
    
    
    
    
    #Random Forest
    pkl_filename = 'final_model2.pkl'
    print("\n Analysis of Random Forest with hypertuned parameters")    
    # rnd_forest = RandomForestClassifier()    
    rnd_forest = RandomForestClassifier(n_estimators=70,max_depth=30,min_samples_split=4,min_samples_leaf=2)
    analysis(rnd_forest,features_normalized,target,X_train,y_train,X_test,y_test,pkl_filename)

    # #Desicion Tree
    # pkl_filename = 'DecisionTree.pkl'
    # print("\n Analysis of Decision Tree with hypertuned parameters")        
    # dn_tree = DecisionTreeClassifier(max_depth=130,min_samples_split=10,min_samples_leaf=2)
    # dn_tree.fit(X_train,y_train)
    # analysis(dn_tree,features_normalized,target,X_train,y_train,X_test,y_test,pkl_filename)

    # Gradient Boost
    pkl_filename = 'final_model1.pkl'
    print("\n Analysis of Gradient Boost with hypertuned parameters")        
    gbc = GradientBoostingClassifier(n_estimators=40,learning_rate=2)
    gbc.fit(X_train,y_train)
    analysis(gbc,features_normalized,target,X_train,y_train,X_test,y_test,pkl_filename)

    
    
    
    
    return
    
    




if __name__ =="__main__":
    main()
