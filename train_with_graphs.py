import sys
import matplotlib.pyplot as plt
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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score #for_accuracy
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from matplotlib.colors import Normalize
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
    return scores.mean()

def accuracy_calculation_graph(classifier_object,x,y,X_train,y_train,X_test,y_test):
    scores = cross_val_score(classifier_object,x,y,cv=5)
    return scores.mean()


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
       
    
    
    
    #Random Forest
    pkl_filename = 'final_model2.pkl'
    print("\n Analysis of Random Forest with hypertuned parameters")    
    # rnd_forest = RandomForestClassifier()    
    rnd_forest = RandomForestClassifier(n_estimators=70,max_depth=30,min_samples_split=4,min_samples_leaf=2)
    analysis(rnd_forest,features_normalized,target,X_train,y_train,X_test,y_test,pkl_filename)
    print("\n plotting graphs....please wait")
    n_estim = [2,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150]
    n_estim2 = [2,4,6,8,10,12,14,16,18,20]
    cv_scores =[]
    cv_scores_max_depth=[]
    for i in n_estim:
        rnd_forest.set_params(n_estimators=i)
        cv_scores.append(accuracy_calculation_graph(rnd_forest,features_normalized,target,X_train,y_train,X_test,y_test))

    for i in n_estim:
        rnd_forest.set_params(max_depth=i)
        cv_scores_max_depth.append(accuracy_calculation_graph(rnd_forest,features_normalized,target,X_train,y_train,X_test,y_test))

    #commented out to save time 
    # for i in n_estim2:
    #     rnd_forest.set_params(min_samples_split=i)
    #     cv_scores_split.append(accuracy_calculation_graph(rnd_forest,features_normalized,target,X_train,y_train,X_test,y_test))

    # for i in n_estim2:
    #     rnd_forest.set_params(min_samples_leaf=i)
    #     cv_scores_leaf.append(accuracy_calculation_graph(rnd_forest,features_normalized,target,X_train,y_train,X_test,y_test))    

    plt.figure(figsize=[16,10])
    plt.style.use('ggplot')
    plt.plot(n_estim, cv_scores,label='Estimators',alpha=0.8)
    plt.plot(n_estim,cv_scores_max_depth,label='Max Depth',alpha=0.8)
    plt.legend()
    plt.xlabel('Number of Estimators/max_depth')
    plt.ylabel('Train Accuracy')
    plt.savefig('Random_forest1.png',dpi=300,bbox_inches='tight')
    plt.show()
    plt.clf()

    #commented out to save time
    # plt.figure(figsize=[16,10])
    # plt.style.use('ggplot')
    # plt.plot(n_estim2, cv_scores_leaf,label='Min Sample Leaf')
    # plt.plot(n_estim2,cv_scores_split,label='Min Sample Split Size')
    # plt.legend()
    # plt.xlabel('Number of Estimators/max_depth')
    # plt.ylabel('Train Accuracy')
    # plt.savefig('Random_forest2.png',dpi=300,bbox_inches='tight')
    # plt.show()
    # plt.clf()
    

    # Gradient Boost
    pkl_filename = 'final_model1.pkl'
    print("\n Analysis of Gradient Boost with hypertuned parameters")        
    gbc = GradientBoostingClassifier(n_estimators=40,learning_rate=2)
    gbc.fit(X_train,y_train)
    analysis(gbc,features_normalized,target,X_train,y_train,X_test,y_test,pkl_filename)
    print("\n plotting graphs....please wait")    
    n_estim = [40,50,60,70,80,90,100]
    learning_rate = [0.1,0.05,0.001,0.5,1.0]
    plt.figure(figsize=[16,10])
    plt.style.use('ggplot')
    for j in learning_rate:
        gbc.set_params(learning_rate=j)
        cv_scores=[]
        for i in n_estim:
            gbc.set_params(n_estimators=i)
            cv_scores.append(accuracy_calculation_graph(gbc,features_normalized,target,X_train,y_train,X_test,y_test))
        
        plt.plot(n_estim, cv_scores,label='Learning_rate'+str(j),alpha=0.8)
    plt.legend()
    plt.xlabel('Estimators and learning Rate Trade Off')
    plt.ylabel('Train Accuracy')
    plt.savefig('GadientBoost3.png',dpi=300,bbox_inches='tight')
    plt.show()
    plt.clf()
    
    
    
    return
    
    




if __name__ =="__main__":
    main()
