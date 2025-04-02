import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, cast, Dict, List, Optional, Union, Callable, OrderedDict

from sklearn.model_selection import train_test_split
from sklearn import metrics 

from time import process_time, process_time_ns

def load_data(path:str) -> pd.DataFrame:
    data = pd.read_csv(path)
    return data

def augment_df(df:pd.DataFrame, disc_cat_num:int = 10) -> pd.DataFrame:
    df = remove_cols(df)
    df = remove_nans(df)
    df = LabelEncoderDf(df)
    
    
    colLabDict = {"cap-shape"           : ["b","c","x","f", "s", "p", "o"],
                  "cap-surface"         : ["i", "g", "y", "s", "d", "h", "l", "k", "t", "w", "e"],
                  "cap-color"           : ["n", "b", "g", "r", "p", "u", "e", "w", "y", "l", "o", "k"],
                  "does-bruise-or-bleed": ["t","f"],
                  "gill-attachment"     : ["a", "x", "d", "e", "s", "p", "f", "?"],
                  "gill-color"          : ["n", "b", "g", "r", "p", "u", "e", "w", "y", "l", "o", "k", "f"],
                  "stem-color"          : ["n", "b", "g", "r", "p", "u", "e", "w", "y", "l", "o", "k", "f"],
                  "ring-type"           : ["c", "e", "r", "g", "l", "p", "s", "z", "y", "m", "f", "?"],
                  "habitat"             : ["g", "l", "m", "p", "h", "u", "w", "d"],
                  "season"              : ["s", "u", "a", "w"],

                  }
    
    df = OrdinalEncoderDf(df, colLabelDict=colLabDict)
    
    df = remove_nans(df)
    
    return df

def LabelEncoderDf(df:pd.DataFrame) -> pd.DataFrame:
    from sklearn.preprocessing import LabelEncoder
    enc = LabelEncoder()
    df["class"] = enc.fit_transform(df["class"])
    return df

def OrdinalEncoderDf(df:pd.DataFrame, colLabelDict) -> pd.DataFrame:
    from sklearn.preprocessing import OrdinalEncoder
    for iter, (key, val) in enumerate(colLabelDict.items()):
        enc = OrdinalEncoder(categories=[val])
        # print(f"key: {key} | val: {val}")
        df[key] = enc.fit_transform(df[[key]])
    
    return df

def OneHotEncodeDf(df:pd.DataFrame) -> pd.DataFrame:
    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder(handle_unknown="ignore")
    
def remove_cols(df:pd.DataFrame) -> pd.DataFrame:
    cols = ["spore-print-color", "veil-type", "veil-color", "stem-root", "stem-surface", "gill-spacing", "has-ring"]
    df.drop(columns=cols, inplace=True)
    return df 

def remove_nans(df:pd.DataFrame) -> pd.DataFrame:
    df.dropna(inplace=True)
    return df

def discretize_by(ser:pd.Series, quart_num:int = 10) -> pd.Series:
    return pd.qcut(ser, quart_num, duplicates='drop')

def discretize(df:pd.DataFrame, disc_cat_num) ->pd.DataFrame:
    df["cap-diameter"] = discretize_by(df["cap-diameter"], disc_cat_num)
    df["stem-height"] = discretize_by(df["stem-height"], disc_cat_num)
    df["stem-width"] = discretize_by(df["stem-width"], disc_cat_num)
    
    labs = {      "cap-diameter"        : df["cap-diameter"].unique(),
                  "stem-height"         : df["stem-height"].unique(),
                  "stem-width"          : df["stem-width"].unique()}
    
    df= OrdinalEncoderDf(df, colLabelDict=labs)
    df = remove_nans(df)

    return df

def NaiveBayes_classifier(train_df:pd.DataFrame, train_labels:pd.Series, test:bool = False, test_df:pd.DataFrame=None, config:list[int] = [10]) -> int:
    from sklearn.naive_bayes import GaussianNB

    nb = GaussianNB(var_smoothing=config[0])
    
    if test == False:
        xtrain, xtest, ytrain, ytest = train_test_split(train_df, train_labels, test_size=0.3)
        
        nb.fit(xtrain, ytrain)
        ypred = nb.predict(xtest)
        acc = metrics.accuracy_score(y_true=ytest, y_pred=ypred)
        
        return "NaiveBayes", acc
    else:
        nb.fit(train_df, train_labels)
        ypred = nb.predict(test_df)
        pred_out = pd.DataFrame(ypred, columns=["class"])

        return "NaiveBayes", pred_out

def Random_Forest_classifier(train_df:pd.DataFrame, train_labels:pd.Series, test:bool = False, test_df:pd.DataFrame=None, config:list[int]=[100, 100]) -> int:
    from sklearn.ensemble import RandomForestClassifier
    rf_model = RandomForestClassifier(n_estimators=config[0], max_depth=config[1], n_jobs=10)
    
    if test == False:
        xtrain, xtest, ytrain, ytest = train_test_split(train_df, train_labels, test_size=0.3)
        
        rf_model.fit(xtrain, ytrain)
        ypred = rf_model.predict(xtest)
        acc = metrics.accuracy_score(y_true=ytest, y_pred=ypred)
        
        return "RandomForest", acc
    else:
        rf_model.fit(train_df, train_labels)
        ypred = rf_model.predict(test_df)
        pred_out = pd.DataFrame(ypred, columns=["class"])
        
        return "RandomForest",pred_out

def KNN_classifier(train_df:pd.DataFrame, train_labels:pd.Series, test:bool = False, test_df:pd.DataFrame=None, config:list[int]=[3]) -> int:
    from sklearn.neighbors import KNeighborsClassifier
    knn_model = KNeighborsClassifier(n_neighbors=config[0], n_jobs=10)
    
    if test == False:
        xtrain, xtest, ytrain, ytest = train_test_split(train_df, train_labels, test_size=0.3)
        
        knn_model.fit(xtrain, ytrain) if isinstance(xtrain, np.ndarray) else knn_model.fit(xtrain.values, ytrain)
        ypred = knn_model.predict(xtest) if isinstance(xtest, np.ndarray) else knn_model.predict(xtest.values) 
        acc = metrics.accuracy_score(y_true=ytest, y_pred=ypred)
        
        return "KNN", acc
    else:
        knn_model.fit(train_df.values, train_labels)
        ypred = knn_model.predict(test_df.values)
        pred_out = pd.DataFrame(ypred, columns=["class"])
        
        return "KNN", pred_out

def Decision_tree_classifier(train_df:pd.DataFrame, train_labels:pd.Series, test:bool=False, 
                             test_df:pd.DataFrame=None, config:list[Union[str,int]]=["entropy", 1, 10, 100]) -> int:
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import cross_val_score
    Dtree = DecisionTreeClassifier(criterion=config[0], min_samples_leaf=config[1], min_samples_split=config[2], max_depth=config[3])
    
    if test == False:
        # scores = cross_val_score(Dtree, train_df, train_labels, cv=3)
        # return "DecisionTree", scores
    
        xtrain, xtest, ytrain, ytest = train_test_split(train_df, train_labels, test_size=0.3)
        
        Dtree.fit(xtrain, ytrain) if isinstance(xtrain, np.ndarray) else Dtree.fit(xtrain.values, ytrain)
        ypred = Dtree.predict(xtest) if isinstance(xtest, np.ndarray) else Dtree.predict(xtest.values) 
        acc = metrics.accuracy_score(y_true=ytest, y_pred=ypred)
        
        return "DecisionTree", acc
    else:
        Dtree = Dtree.fit(train_df, train_labels)
        ypred = Dtree.predict(test_df)
        
        pred_df = pd.DataFrame(ypred, columns=["Label"])

        return pred_df

def Kmeans_classifier(train_df:pd.DataFrame, train_labels:pd.Series, test:bool = False, test_df:pd.DataFrame=None, config:list[Union[int, str]] = [2]) -> int:
    from sklearn.cluster import KMeans
    kmeans_model = KMeans(n_clusters=config[0])
    
    if test == False:
        
        xtrain, xtest, ytrain, ytest = train_test_split(train_df, train_labels, test_size=0.3)
        kmeans_model.fit(xtrain, ytrain)
        ypred = kmeans_model.predict(xtest)
        acc = metrics.accuracy_score(y_true=ytest, y_pred=ypred)
        
        return "Kmeans", acc
    else:
        kmeans_model.fit(train_df, train_labels)
        ypred = kmeans_model.predict(test_df)
        pred_out = pd.DataFrame(ypred, columns=["class"])
        
        return "Kmeans", pred_out
    
def linear_classifier(train_df:pd.DataFrame, train_labels:pd.Series, test:bool = False, test_df:pd.DataFrame=None, config:list[Union[int, str]] = [True]) -> int:
    from sklearn.linear_model import LinearRegression
    linear_model = LinearRegression(fit_intercept=config[0])
    
    if test == False:
        
        xtrain, xtest, ytrain, ytest = train_test_split(train_df, train_labels, test_size=0.3)
        linear_model.fit(xtrain, ytrain)
        ypred = linear_model.predict(xtest)
        acc = metrics.accuracy_score(y_true=ytest, y_pred=ypred)
        
        return "linear", acc
    else:
        linear_model.fit(train_df, train_labels)
        ypred = linear_model.predict(test_df)
        pred_out = pd.DataFrame(ypred, columns=["Label"])
        
        return "linear", pred_out
    
def SVM_classifier(train_df:pd.DataFrame, train_labels:pd.Series, test:bool = False, test_df:pd.DataFrame=None, config:list[Union[int, str]] = [1, 'linear']) -> int:
    from sklearn import svm
    svm_model = svm.SVC(C=config[0], kernel=config[1])
    
    if test == False:
        
        xtrain, xtest, ytrain, ytest = train_test_split(train_df, train_labels, test_size=0.3)
        svm_model.fit(xtrain, ytrain)
        ypred = svm_model.predict(xtest)
        acc = metrics.accuracy_score(y_true=ytest, y_pred=ypred)
        
        return "SVM", acc
    else:
        svm_model.fit(train_df, train_labels)
        ypred = svm_model.predict(test_df)
        pred_out = pd.DataFrame(ypred, columns=["Label"])
        
        return "SVM", pred_out

def hyperparameter_crawler(df:pd.DataFrame) -> plt.plot:
    RFconfigs = {"c_estimators":{1:[1, 10], 2:[10,10], 3:[100,10], 4:[1000,10]},
                 "c_depth":{1:[100, 1], 2:[100, 10], 3:[100, 100], 4:[100, 1000],}}
    NBconfigs = {"c_varsmoothing":{1:[1], 2:[5], 3:[10], 4:[50], 5:[100]}}
    DTconfigs = {
                 "c_loss":{1:["log_loss", 1, 10, 100], 2:["gini", 1, 10, 100], 3:["entropy", 1, 10, 100], },
                 "c_leaf":{1:["log_loss", 1, 10, 100], 2:["gini", 5, 10, 100], 3:["entropy", 10, 10, 100], 4:["entropy", 25, 10, 100], 5:["entropy", 100, 10, 100], },
                 "c_split":{1:["log_loss", 1, 2, 100], 2:["gini", 1, 5, 100],  3:["entropy", 1, 10, 100],  4:["entropy", 1, 25, 100],  5:["entropy", 1, 100, 100], },
                 "c_depth":{1:["log_loss", 1, 10, 1],  2:["gini", 1, 10, 5],   3:["entropy", 1, 10, 10],   4:["entropy", 1, 10, 25],   5:["entropy", 1, 10, 100], }}
    
    KNNconfigs = {"c_neighbors":{1:[1], 2:[5], 3:[10], 4:[25], 5:[50], 6:[100]}}
    Kmeansconfigs = {"c_clusters":{1:[1], 2:[5], 3:[10], 4:[25], 5:[50]}}
    linearconfigs = {"c_intercept":{1:[True], 2:[False]}}
    
    
    train_labels = df["class"]
    df.drop(columns=["class"], inplace=True)
    
    # print("starting Random Forest")
    
    # # c_estimators |   rt | acc
    # #                  rt | acc
    # #                  rt | acc
    # # c_depth      |   rt | acc
    # #                  rt | acc
    # #                  rt | acc
    
    # RFaccs = {"c_estimators": {"rt":[], "acc":[], "c_val":[]},
    #           "c_depth":{"rt":[], "acc":[], "c_val":[]}}
    # counter = 0
    # for key in RFconfigs:
    #     for id, (val, config) in enumerate(RFconfigs[key].items()):
    #         # print(f"key: {key} | id: {id} | val: {val} | config: {config}")
    #         start = process_time()
    #         model_type, acc = Random_Forest_classifier(train_df=df, train_labels=train_labels, config=config, test=False)
    #         end = process_time() - start
            
    #         RFaccs[key]["rt"].append(end)
    #         RFaccs[key]["acc"].append(acc)
    #         RFaccs[key]["c_val"].append(config[counter])
    #     counter += 1
    
    # print("plotting")
    # make_accrt_plot(RFaccs,  "Random Forest")
    
    # print("starting Naive Bayes")
    # # c_estimators |   rt | acc
    # #                  rt | acc
    # #                  rt | acc
    # # c_depth      |   rt | acc
    # #                  rt | acc
    # #                  rt | acc
    
    # NBaccs = {"c_varsmoothing": {"rt":[], "acc":[], "c_val":[]}}
    # counter = 0
    # for key in NBconfigs:
    #     for id, (val, config) in enumerate(NBconfigs[key].items()):
    #         # print(f"key: {key} | id: {id} | val: {val} | config: {config}")
    #         start = process_time()
    #         model_type, acc = NaiveBayes_classifier(train_df=df, train_labels=train_labels, config=config, test=False)
    #         end = process_time() - start
            
    #         # NBaccs[key].update({config:[end, acc]})
            
    #         NBaccs[key]["rt"].append(end)
    #         NBaccs[key]["acc"].append(acc)
    #         NBaccs[key]["c_val"].append(config[counter])
    #     counter += 1

    # print("plotting")
    # make_accrt_plot(NBaccs,  "Naive Bayes")   
    # print("starting Decision Tree")
    # # c_estimators |   rt | acc
    # #                  rt | acc
    # #                  rt | acc
    # # c_depth      |   rt | acc
    # #                  rt | acc
    # #                  rt | acc
    
    # DTaccs = {"c_loss": {"rt":[], "acc":[], "c_val":[]},
    #           "c_leaf": {"rt":[], "acc":[], "c_val":[]},
    #           "c_split":{"rt":[], "acc":[], "c_val":[]},
    #           "c_depth":{"rt":[], "acc":[], "c_val":[]}}
    # counter = 0
    # for key in DTconfigs:
    #     for id, (val, config) in enumerate(DTconfigs[key].items()):
    #         # print(f"key: {key} | id: {id} | val: {val} | config: {config}")
    #         start = process_time()
    #         model_type, acc = Decision_tree_classifier(train_df=df, train_labels=train_labels, config=config, test=False)
    #         end = process_time() - start
            
    #         # DTaccs[key].update({config:[end, acc]})
            
    #         DTaccs[key]["rt"].append(end)
    #         # print(DTaccs[key]['rt'])
    #         DTaccs[key]["acc"].append(sum(acc)/3)
    #         # print(DTaccs[key]['acc'])
    #         DTaccs[key]["c_val"].append(config[counter])
    #     counter += 1
            
    

    # print("plotting")
    # make_accrt_plot(DTaccs,  "Decision Tree")
    # print("starting KNN") 
    # # c_estimators |   rt | acc
    # #                  rt | acc
    # #                  rt | acc
    # # c_depth      |   rt | acc
    # #                  rt | acc
    # #                  rt | acc
    
    # KNNaccs = {"c_neighbors": {"rt":[], "acc":[], "c_val":[]}}
    # counter = 0
    # for key in KNNconfigs:
    #     for id, (val, config) in enumerate(KNNconfigs[key].items()):
    #         # print(f"key: {key} | id: {id} | val: {val} | config: {config}")
    #         start = process_time()
    #         model_type, acc = KNN_classifier(train_df=df, train_labels=train_labels, config=config, test=False)
    #         end = process_time() - start
            
    #         # KNNaccs[key].update({config:[end, acc]})
    
    #         KNNaccs[key]["rt"].append(end)
    #         KNNaccs[key]["acc"].append(acc)
    #         KNNaccs[key]["c_val"].append(config[counter])
    #     counter += 1


    # print("plotting")
    # make_accrt_plot(KNNaccs, "KNN")
    print("starting Kmeans") 
    # c_estimators |   rt | acc
    #                  rt | acc
    #                  rt | acc
    # c_depth      |   rt | acc
    #                  rt | acc
    #                  rt | acc
    
    KNNaccs = {"c_clusters": {"rt":[], "acc":[], "c_val":[]}}
    counter = 0
    for key in Kmeansconfigs:
        for id, (val, config) in enumerate(Kmeansconfigs[key].items()):
            # print(f"key: {key} | id: {id} | val: {val} | config: {config}")
            start = process_time()
            model_type, acc = Kmeans_classifier(train_df=df, train_labels=train_labels, config=config, test=False)
            end = process_time() - start
            
            # KNNaccs[key].update({config:[end, acc]})
    
            KNNaccs[key]["rt"].append(end)
            KNNaccs[key]["acc"].append(acc)
            KNNaccs[key]["c_val"].append(config[counter])
        counter += 1


    print("plotting")
    make_accrt_plot(KNNaccs, "Kmeans")
    
    # print("starting linear") 
    # # c_estimators |   rt | acc
    # #                  rt | acc
    # #                  rt | acc
    # # c_depth      |   rt | acc
    # #                  rt | acc
    # #                  rt | acc
    
    # KNNaccs = {"c_intercept": {"rt":[], "acc":[], "c_val":[]}}
    # counter = 0
    # for key in linearconfigs:
    #     for id, (val, config) in enumerate(linearconfigs[key].items()):
    #         # print(f"key: {key} | id: {id} | val: {val} | config: {config}")
    #         start = process_time()
    #         model_type, acc = linear_classifier(train_df=df, train_labels=train_labels, config=config, test=False)
    #         end = process_time() - start
            
    #         # KNNaccs[key].update({config:[end, acc]})
    
    #         KNNaccs[key]["rt"].append(end)
    #         KNNaccs[key]["acc"].append(acc)
    #         KNNaccs[key]["c_val"].append(config[counter])
    #     counter += 1


    # print("plotting")
    # make_accrt_plot(KNNaccs, "Linear")
    
    
    # # model_config_key1      |   rt | acc | c_val | disc_cat_num
    # #                            rt | acc | c_val | disc_cat_num
    # #                            rt | acc | c_val | disc_cat_num
    # # model_config_key1      |   rt | acc | c_val | disc_cat_num
    # #                            rt | acc | c_val | disc_cat_num
    # #                            rt | acc | c_val | disc_cat_num

def hyperparameter_crawler_v2(df:pd.DataFrame):
    RFconfigs = {"c_estimators":{1:[1, 10], 2:[10,10], 3:[100,10], 4:[1000,10]},
                 "c_depth":{1:[100, 1], 2:[100, 10], 3:[100, 100], 4:[100, 1000],}}
    NBconfigs = {"c_varsmoothing":{1:[1], 2:[5], 3:[10], 4:[50], 5:[100]}}
    DTconfigs = {
                 "c_loss":{1:["log_loss", 1, 10, 100], 2:["gini", 1, 10, 100], 3:["entropy", 1, 10, 100], },
                 "c_leaf":{1:["log_loss", 1, 10, 100], 2:["gini", 5, 10, 100], 3:["entropy", 10, 10, 100], 4:["entropy", 25, 10, 100], 5:["entropy", 100, 10, 100], },
                 "c_split":{1:["log_loss", 1, 2, 100], 2:["gini", 1, 5, 100],  3:["entropy", 1, 10, 100],  4:["entropy", 1, 25, 100],  5:["entropy", 1, 100, 100], },
                 "c_depth":{1:["log_loss", 1, 10, 1],  2:["gini", 1, 10, 5],   3:["entropy", 1, 10, 10],   4:["entropy", 1, 10, 25],   5:["entropy", 1, 10, 100], }}
    # 
    KNNconfigs = {"c_neighbors":{1:[1], 2:[5], 3:[10], 4:[25], 5:[50], 6:[100]}}
    Kmeansconfigs = {"c_clusters":{1:[1], 2:[5], 3:[10], 4:[25], 5:[50]}}
    # linearconfigs = {"c_intercept":{1:[True], 2:[False]}}
    
    disc_configs = [4, 10, 15, 35, 55, 115]
    #Kmeans_classifier, KNN_classifier, Random_Forest_classifier, 
    model_functions = [Decision_tree_classifier]
    #Kmeansconfigs, KNNconfigs, RFconfigs, 
    model_config_list = [DTconfigs]
    
    model_to_config_dict = dict(zip(model_functions, model_config_list))
    
    for idx, (model_function, model_config) in enumerate(model_to_config_dict.items()):
        print(model_config)
        crawl_model(model_config, disc_configs, df, model_function)

def crawl_model(model_configs:list[Union[list[int,str]]], disc_cat_configs:list[int], df:pd.DataFrame, model_func) -> dict[str:list[Union[str,int]]]:
    import json
    result_dict = {config: {"rt":[], "acc":[], "c_val":[], "disc_cat_num":[]} for config in model_configs.keys()}
    print(result_dict)
    counter = 0
    for key in model_configs:
        print(f"starting crawl for config number: {counter}")
        for id, (val, config) in enumerate(model_configs[key].items()):
            for disc_cat in disc_cat_configs:
                aug_df2 = df.copy(deep=True)
                # print(df.columns)
                aug_df2 = discretize(aug_df2, disc_cat)
                print(len(aug_df2))
                # print(aug_df_disced.columns)
                train_labels = aug_df2["class"]
                aug_df2 = aug_df2.drop(columns=["class"])
                
                # print(f"key: {key} | id: {id} | val: {val} | config: {config}")
                start = process_time()
                model_type, acc = model_func(train_df=aug_df2, train_labels=train_labels, config=config, test=False)
                end = process_time() - start

                result_dict[key]["rt"].append(end)
                result_dict[key]["acc"].append(acc)
                result_dict[key]["c_val"].append(config[counter])
                result_dict[key]["disc_cat_num"].append(disc_cat)
        counter += 1

    print(json.dumps(result_dict, sort_keys=True, indent=4))
    
    print("plotting")
    make_accrt_plot(result_dict, model_type)
    
    with open("/Users/Shintaro/Documents/Syracuse_MS/Summer_2024/IST707/Project/Model_outputs/"+model_type+"outputs.json", "w") as f:
        json.dump(result_dict, f)
        
    
    # return result_dict 


def make_accrt_plot(accDict:Dict, model_name:str):
    for key in accDict:
        if key == "c_loss":
            plt.plot(accDict[key]["acc"])
        else:
            plt.plot(accDict[key]["c_val"], accDict[key]["acc"])
        plt.title(model_name+key+" acc plot")
        plt.savefig("/Users/Shintaro/Documents/Syracuse_MS/Summer_2024/IST707/Project/Paper_plots/"+model_name+key+'_acc.png')
        # plt.show()
        plt.cla()
        plt.plot(accDict[key]["rt"])
        plt.title(model_name+" "+key+" rt plot")
        plt.savefig("/Users/Shintaro/Documents/Syracuse_MS/Summer_2024/IST707/Project/Paper_plots/"+model_name+key+'_rt.png')
        # plt.show()
        plt.cla()
        # model config acc by disc cat
        plt.plot(accDict[key]["disc_cat_num"], accDict[key]["acc"])
        plt.title(model_name+key+" cat disc by acc plot")
        plt.savefig("/Users/Shintaro/Documents/Syracuse_MS/Summer_2024/IST707/Project/Paper_plots/"+model_name+key+'_disc.png')
        # plt.show()
        plt.cla()

def make_cat_acc_plot(df:pd.DataFrame, model_name:str):
    for name, group in df.groupby("c_val"):
        plt.plot(group["disc_cat_num"], group["acc"], label=f'C_val={name}')

    # Set plot labels and title
    plt.xlabel("Category Num")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Category Num grouped by c_val")
    plt.legend()

    # Display the plot
    # plt.show()
    plt.savefig("/Users/Shintaro/Documents/Syracuse_MS/Summer_2024/IST707/Project/Paper_plots/"+model_name+"_disc.png")
    plt.cla()
    
def make_cat_rt_plot(df:pd.DataFrame, model_name:str):
    for name, group in df.groupby("c_val"):
        plt.plot(group["disc_cat_num"], group["rt"], label=f'C_val={name}')

    # Set plot labels and title
    plt.xlabel("Category Num")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Category Num grouped by c_val")
    plt.legend()

    # Display the plot
    # plt.show()
    plt.savefig("/Users/Shintaro/Documents/Syracuse_MS/Summer_2024/IST707/Project/Paper_plots/"+model_name+"_rt.png")
    plt.cla()
    
    
def plot_cat_acc():
    DT_outputs_loss = pd.DataFrame({"rt": [0.75, 0.890625, 1.5, 0.28125, 2.515625, 1.78125, 1.265625, 1.53125, 1.484375, 3.234375, 2.15625, 0.859375, 0.703125, 0.84375, 0.828125, 1.4375, 0.765625, 0.765625],                       
                       "acc": [0.998988455968396, 0.9996475541813894, 0.9997677619199874, 0.9998528068506962, 0.9998593487684431, 0.9998552600698514, 0.9989794608314941, 0.9996524606196995, 0.9997710328788608, 0.9998593487684431, 0.9998609842478798, 0.9998078311661868, 0.9989622882974086, 0.999646736441671, 0.9997800280157627, 0.9998560778095696, 0.999842993974076, 0.999829092398864], 
                       "c_val": ["log_loss", "log_loss", "log_loss", "log_loss", "log_loss", "log_loss", "gini", "gini", "gini", "gini", "gini", "gini", "entropy", "entropy", "entropy", "entropy", "entropy", "entropy"], 
                       "disc_cat_num": [4, 10, 15, 35, 55, 115, 4, 10, 15, 35, 55, 115, 4, 10, 15, 35, 55, 115]})
    
    DT_outputs_leaf = pd.DataFrame({"rt": [0.96875, 0.9375, 1.375, 2.046875, 1.390625, 2.28125, 1.75, 0.234375, 1.5625, 0.71875, 0.171875, 0.4375, 1.125, 0.609375, 1.375, 2.15625, 1.125, 1.265625, 1.03125, 0.625, 1.171875, 0.765625, 1.078125, 0.40625, 0.296875, 1.390625, 1.96875, 2.203125, 3.015625, 1.28125], 
                       "acc": [0.9990064462421998, 0.9996524606196995, 0.9997407765092817, 0.9998675261656267, 0.9998642552067533, 0.9998299101385824, 0.9989802785712124, 0.9995919478805413, 0.9997358700709716, 0.9998299101385824, 0.9998029247278767, 0.9997514071256204, 0.9989385738455764, 0.999579681784766, 0.9996606380168831, 0.999775939317171, 0.9997669441802691, 0.9997391410298451, 0.9989164948731808, 0.9994529321284211, 0.9995109916484243, 0.9995984897982881, 0.9996230219898388, 0.9996336526061773, 0.9986793503548581, 0.9991749006241807, 0.9988935981610669, 0.9989712834343105, 0.9990571461047377, 0.9990260719954402], 
                       "c_val": [1, 1, 1, 1, 1, 1, 5, 5, 5, 5, 5, 5, 10, 10, 10, 10, 10, 10, 25, 25, 25, 25, 25, 25, 100, 100, 100, 100, 100, 100], 
                       "disc_cat_num": [4, 10, 15, 35, 55, 115, 4, 10, 15, 35, 55, 115, 4, 10, 15, 35, 55, 115, 4, 10, 15, 35, 55, 115, 4, 10, 15, 35, 55, 115]})

    DT_outputs_split = pd.DataFrame({"rt": [0.3125, 0.828125, 0.609375, 1.8125, 0.34375, 0.453125, 0.359375, 1.421875, 1.828125, 1.4375, 1.734375, 2.453125, 1.0, 1.234375, 0.5, 0.015625, 0.40625, 0.390625, 0.71875, 1.21875, 0.9375, 1.0625, 2.28125, 1.796875, 1.375, 1.875, 0.40625, 1.03125, 0.234375, 0.546875], 
                        "acc": [0.9989737366534657, 0.9996516428799812, 0.9997628554816773, 0.9998495358918228, 0.999856895549288, 0.9998307278783007, 0.9989802785712124, 0.999649189660826, 0.9997620377419589, 0.9998642552067533, 0.9998511713712596, 0.9998078311661868, 0.9990203478174118, 0.9996565493182913, 0.9997579490433672, 0.9998536245904146, 0.9998470826726678, 0.9998258214399905, 0.9990088994613548, 0.9996271106884306, 0.999778392536326, 0.9998217327413988, 0.9998323633577374, 0.999812737604497, 0.9989148593937441, 0.9995690511684274, 0.9997056137013925, 0.9997407765092817, 0.9997293281532248, 0.9997015250028007], 
                        "c_val": [2, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5, 5, 10, 10, 10, 10, 10, 10, 25, 25, 25, 25, 25, 25, 100, 100, 100, 100, 100, 100], 
                        "disc_cat_num": [4, 10, 15, 35, 55, 115, 4, 10, 15, 35, 55, 115, 4, 10, 15, 35, 55, 115, 4, 10, 15, 35, 55, 115, 4, 10, 15, 35, 55, 115]})

    DT_outputs_depth = pd.DataFrame({"rt": [0.40625, 0.65625, 0.171875, 0.09375, 0.109375, 0.21875, 0.859375, 0.265625, 0.1875, 0.65625, 0.5, 0.28125, 0.84375, 1.234375, 1.390625, 0.828125, 1.53125, 0.65625, 1.4375, 0.6875, 0.609375, 0.671875, 1.890625, 1.703125, 0.828125, 0.78125, 0.4375, 0.25, 1.625, 1.59375], 
                        "acc": [0.6059549441769981, 0.6003092691614815, 0.6178309781066545, 0.6150351260096019, 0.6007573905271395, 0.600811361348551, 0.8300417946770051, 0.8051031864863605, 0.7888367080088611, 0.7990020304477207, 0.7894753627288956, 0.7855117783140333, 0.9686617607735164, 0.9685464594732285, 0.9231831663372538, 0.9069755651194759, 0.9631281160994143, 0.9561127270556545, 0.998977007612339, 0.9996483719211078, 0.999748136166747, 0.999842993974076, 0.9998511713712596, 0.9998364520563292, 0.9990121704202283, 0.9996573670580097, 0.9997914763718198, 0.999851989110978, 0.9998724326039368, 0.9998470826726678], 
                        "c_val": [1, 1, 1, 1, 1, 1, 5, 5, 5, 5, 5, 5, 10, 10, 10, 10, 10, 10, 25, 25, 25, 25, 25, 25, 100, 100, 100, 100, 100, 100], 
                        "disc_cat_num": [4, 10, 15, 35, 55, 115, 4, 10, 15, 35, 55, 115, 4, 10, 15, 35, 55, 115, 4, 10, 15, 35, 55, 115, 4, 10, 15, 35, 55, 115]})

    
    NB_output = pd.DataFrame({"rt": [0.59375, 0.421875, 0.6875, 0.546875, 0.90625, 0.75, 0.6875, 0.703125, 0.671875, 0.65625, 0.46875, 0.546875, 0.671875, 0.78125, 0.5625, 0.578125, 0.5, 0.53125, 0.484375, 0.6875, 0.859375, 0.734375, 0.6875, 0.703125, 0.671875, 0.8125, 0.671875, 0.5, 0.453125, 0.640625], 
        "acc": [0.63348333405567, 0.6338496814494927, 0.6259822076192081, 0.5810171537260719, 0.5724406995599742, 0.563968916077826, 
        0.5411654262918039, 0.5733344890721352, 0.5692539678775483, 0.5317785920648174, 0.5251393632915005, 0.5297227944128752, 
        0.5345965231342655, 0.534288235260446, 0.536322771679711, 0.535156674841338, 0.5345711732029965, 0.5351395023072526, 
        0.5348115886801926, 0.534500847587218, 0.5346562181337053, 0.534722455050892, 0.534747804982161, 0.5349980333359774, 
        0.5346676664897623, 0.5343078610136865, 0.5347576178587813, 0.5354281644278316, 0.5344575073821453, 0.5349636882678065], 
        "c_val": [1, 1, 1, 1, 1, 1, 5, 5, 5, 5, 5, 5, 10, 10, 10, 10, 10, 10, 50, 50, 50, 50, 50, 50, 100, 100, 100, 100, 100, 100], 
        "disc_cat_num": [4, 10, 15, 35, 55, 115, 4, 10, 15, 35, 55, 115, 4, 10, 15, 35, 55, 115, 4, 10, 15, 35, 55, 115, 4, 10, 15, 35, 55, 115]})

    
    RF_output_est = pd.DataFrame({"rt": [0.453125, 0.421875, 0.265625, 0.40625, 0.28125, 0.40625, 14.21875, 17.796875, 19.09375, 16.421875, 17.390625, 18.0625, 147.4375, 134.453125, 163.171875, 150.0625, 188.5, 216.84375, 1631.75, 1854.375, 2090.34375, 2126.546875, 2155.40625, 2174.75], 
"acc": [0.9391454456395256, 0.9165455730433737, 0.871874905448845, 0.8701347553281875, 0.9455933233187476, 0.9170771038603039, 0.9840491690537851, 0.9915691035037694, 0.9860444539665691, 0.9903653906383522, 0.9816106692136534, 0.9869946675192967, 0.992067106992247, 0.9954026673034133, 0.9951344486757931, 0.9953078094960842, 0.9934728015680977, 0.9909754244682443, 0.9917122079544813, 0.9956659794927234, 0.9952252177845304, 0.9953462432628469, 0.9942169447117999, 0.9922731774012722], 
"c_val": [1, 1, 1, 1, 1, 1, 10, 10, 10, 10, 10, 10, 100, 100, 100, 100, 100, 100, 1000, 1000, 1000, 1000, 1000, 1000], 
"disc_cat_num": [4, 10, 15, 35, 55, 115, 4, 10, 15, 35, 55, 115, 4, 10, 15, 35, 55, 115, 4, 10, 15, 35, 55, 115]})

    
    RF_output_depth = pd.DataFrame({"rt": [37.125, 37.953125, 38.03125, 40.15625, 39.859375, 39.75, 191.03125, 198.3125, 204.46875, 211.171875, 213.9375, 208.859375, 213.296875, 225.140625, 237.078125, 237.78125, 247.171875, 250.921875, 207.9375, 221.609375, 234.9375, 243.03125, 238.265625, 251.9375], 
"acc": [0.6335029598089106, 0.6451835539458803, 0.6663989931988588, 0.6577849230057168, 0.6499117249974037, 0.613790526158267, 0.9920842795263325, 0.9950731181969167, 0.9961312733924668, 0.9944050248470213, 0.9927613680131296, 0.9934490871162655, 0.998963106037127, 0.9996737218523767, 0.9998045602073133, 0.9998969647954874, 0.9998904228777405, 0.9998528068506962, 0.999030160694032, 0.999685987948152, 0.9998143730839336, 0.9999035067132342, 0.9998945115763324, 0.9998601665081615], 
"c_val": [1, 1, 1, 1, 1, 1, 10, 10, 10, 10, 10, 10, 100, 100, 100, 100, 100, 100, 1000, 1000, 1000, 1000, 1000, 1000], 
"disc_cat_num": [4, 10, 15, 35, 55, 115, 4, 10, 15, 35, 55, 115, 4, 10, 15, 35, 55, 115, 4, 10, 15, 35, 55, 115]})
    

    KNN_output_neigh = pd.DataFrame({"rt": [731.796875, 458.171875, 330.703125, 202.84375, 132.09375, 100.75, 792.734375, 486.515625, 387.1875, 231.78125, 185.234375, 157.890625, 782.90625, 486.21875, 375.578125, 262.359375, 239.890625, 200.71875, 874.203125, 582.484375, 451.59375, 351.40625, 317.890625, 300.296875,857.859375, 642.515625, 576.015625, 458.078125, 426.09375, 402.390625, 941.328125, 752.765625, 690.796875, 446.828125, 542.484375, 409.34375], 
"acc": [0.9987218728202125, 0.9993678871977123, 0.9996990717836457, 0.9995093561689875, 0.9989164948731808, 0.9945235971061827, 
0.9983637028235735, 0.9996451009622344, 0.9996377413047691, 0.9992288714455921, 0.9980562326894723, 0.9908781134417601, 
0.999030160694032, 0.9995862237025128, 0.999524075483918, 0.9987292324776778, 0.9970512305756152, 0.9874583259396034, 
0.9989786430917758, 0.9994979078129306, 0.9992435907605225, 0.997752851253963, 0.9947272142960528, 0.9809106840147422, 
0.998965559256282, 0.9994398482929274, 0.9987578533678202, 0.9962065054465554, 0.9918741204187155, 0.973951719011549, 
0.9989303964483929, 0.9990726831593865, 0.9979924489914407, 0.9934024759523192, 0.9877927814844102, 0.964707171495556], 
"c_val": [1, 1, 1, 1, 1, 1, 5, 5, 5, 5, 5, 5, 10, 10, 10, 10, 10, 10, 25, 25, 25, 25, 25, 25, 50, 50, 50, 50, 50, 50, 100, 100, 100, 100, 100, 100], 
"disc_cat_num": [4, 10, 15, 35, 55, 115, 4, 10, 15, 35, 55, 115, 4, 10, 15, 35, 55, 115, 4, 10, 15, 35, 55, 115, 4, 10, 15, 35, 55, 115, 4, 10, 15, 35, 55, 115]})

    Kmean_output_clust = pd.DataFrame({"rt": [8.609375, 8.640625, 9.296875, 9.390625, 8.765625, 8.203125, 
19.125, 22.203125, 20.546875, 20.015625, 19.875, 19.625, 
22.84375, 24.578125, 26.234375, 26.859375, 22.8125, 26.671875, 
35.328125, 36.796875, 38.609375, 47.234375, 40.125, 42.859375, 
49.546875, 54.109375, 60.09375, 70.0625, 71.484375, 69.75], 
"acc": [0.46492918782908915, 0.46521866768938647, 0.4647860833783772, 0.4653061658392504, 0.465081287416703, 0.4655580296725034, 0.2212313034035145, 0.16262553326851384, 0.16557675591205373, 0.1939204322899247, 0.1826339886972016, 0.2820547836546914, 0.06654275184134541, 0.08143215663313662, 0.0909342921604111, 0.1261003710085102, 0.08478161851951495, 0.09531247061247887, 0.05098688917909563, 0.029232559451721873, 0.07006802776716987, 0.03959741038185992, 0.0635686324856916, 0.03465499152412782, 0.012718305839561102, 0.01562618827802823, 0.013715130556234734, 0.006840392744031931, 0.015200963624484108, 0.019278213860197582],
 "c_val": [1, 1, 1, 1, 1, 1, 5, 5, 5, 5, 5, 5, 10, 10, 10, 10, 10, 10, 25, 25, 25, 25, 25, 25, 50, 50, 50, 50, 50, 50], 
 "disc_cat_num": [4, 10, 15, 35, 55, 115, 4, 10, 15, 35, 55, 115, 4, 10, 15, 35, 55, 115, 4, 10, 15, 35, 55, 115, 4, 10, 15, 35, 55, 115]})

    df_list = [DT_outputs_depth, DT_outputs_leaf, DT_outputs_loss, DT_outputs_split, NB_output, Kmean_output_clust, KNN_output_neigh, RF_output_depth, RF_output_est]
    df_names = ["DT_depth", "DT_leaf", "DT_loss", "DT_split", "NB_var", "Kmean_clust", "KNN_neigh", "RF_depth", "RF_est"]
    
    for (val, names) in zip(df_list, df_names):
        print(names)
        make_cat_acc_plot(val, names)
        make_cat_rt_plot(val, names)
    

def make_all_hist_sep(df:pd.DataFrame):
    for idx, key in enumerate(df.columns):
        print(key)
        df[key].hist()
        plt.title("Histogram of " +key)
        plt.savefig("/Users/Shintaro/Documents/Syracuse_MS/Summer_2024/IST707/Project/Paper_plots/"+key+"_hist.png")
        plt.cla()
        # break
    
def make_plots(df:pd.DataFrame) -> None:
    make_hist_dist(df)

           
def make_hist_dist(df:pd.DataFrame) -> None:
    col_list = ["cap-shape", "cap-surface", "cap-color", "does-bruise-or-bleed", "gill-attachment", "gill-color", "stem-color", "ring-type", "season"]
    fig1, axs1 = plt.subplots(ncols=4, nrows=3)
    # fig2, axs2 = plt.subplots(ncols=4, nrows=3)
    for idx, col in enumerate(col_list):
        print(col)
        # df.groupby("season")[col].hist(alpha=0.4, ax=axs[idx//3, idx%3])
        # df.groupby("class").plot.pie(y=col, ax=axs[idx//4, idx%3], title=col)
        df.groupby("habitat")[col].plot.kde(ax=axs1[idx//4, idx%4], title=col, legend=True)
        # df.groupby("season")[col].plot.kde(ax=axs2[idx//4, idx%4], title=col)
        # df.plot.pie(y=col, labels=df['class'], autopct='%1.1f', ax=axs[idx//4, idx%3])
        
        
    # df.hist(column=["cap-shape", "cap-surface", "cap-color", "does-bruise-or-bleed", "gill-attachment", "gill-color", "stem-color", "ring-type", "habitat"], by="season")
    # fig, axs = plt.subplots(ncols=2)
    plt.show()

def make_stacked_bar_all(df: pd.DataFrame, group_var: str, normalize: bool = False) -> None:
    # Step 1: Loop through all columns except the group_var
    col_list = [col for col in df.columns if col != group_var]
    
    # Step 2: Set up subplots - nrows based on number of variables to plot
    num_cols = len(col_list)
    ncols = 3
    nrows = (num_cols + ncols - 1) // ncols  # Calculate the number of rows needed
    
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5 * nrows))
    axs = axs.flatten()  # Flatten for easy access to axes
    
    for idx, x_var in enumerate(col_list):
        # Step 3: Group by the current column (x_var) and the group_var, count occurrences
        group_sizes = df.groupby([x_var, group_var]).size().unstack(fill_value=0)
        
        # Step 4: Normalize if required
        if normalize:
            group_sizes = group_sizes.div(group_sizes.sum(axis=1), axis=0)
        
        # Step 5: Plot on the corresponding subplot
        group_sizes.plot(kind='bar', stacked=True, ax=axs[idx], colormap='Set2', legend=True)
        axs[idx].set_title(f'{x_var} grouped by {group_var}')
        axs[idx].set_ylabel('Proportion' if normalize else 'Count')
        axs[idx].set_xlabel(x_var)
        axs[idx].tick_params(axis='x', rotation=45)
    
    # Step 6: Adjust layout and show plot
    plt.tight_layout()
    handles, labels = axs[0].get_legend_handles_labels()  # Fetch handles/labels from the first plot
    fig.legend(handles, labels, title=group_var, loc='upper right', bbox_to_anchor=(1.1, 1))  # Global legend
    
    # plt.legend(title=group_var, loc='upper right', bbox_to_anchor=(1.2, 1))
    plt.savefig(group_var +" stacked_bar_plot")
    plt.show()
    

def make_stacked_bar(df: pd.DataFrame, x_var: str, group_var: str, normalize: bool = False) -> None:
    # Step 1: Group by the two selected variables and count occurrences
    group_sizes = df.groupby([x_var, group_var]).size().unstack(fill_value=0)

    # Step 2: Normalize if required (to show proportions instead of raw counts)
    if normalize:
        group_sizes = group_sizes.div(group_sizes.sum(axis=1), axis=0)
    
    # Step 3: Plot the stacked bar plot
    group_sizes.plot(kind='bar', stacked=True, colormap='Set2', figsize=(10, 6))

    # Step 4: Customize the plot
    plt.title(f'Stacked Bar Plot of {x_var} grouped by {group_var}')
    plt.ylabel('Proportion' if normalize else 'Count')
    plt.xlabel(x_var)
    plt.legend(title=group_var)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def make_stacked_hist(df:pd.DataFrame) -> None:
    df.groupby("habitat").plot(kind="bar", stacked=True, legend=True, colormap="Set2")
    plt.show()
    
def print_nan(df:pd.DataFrame) -> None:
    print(df[df['cap-surface'].isna()])
    print(len(df[df['cap-surface'].isna()]))

def train_models(aug_df:pd.DataFrame) -> None:

    # print(aug_df)
    # print(aug_df.columns)
    train_labels = aug_df["class"]
    aug_df.drop(columns=["class"], inplace=True)
    start = process_time()
    model_type, acc = Random_Forest_classifier(train_df=aug_df, train_labels=train_labels, test=False)
    end = process_time() - start
    print(f"acc: {acc} | rt: {end}")
    start = process_time()
    model_type, acc = NaiveBayes_classifier(train_df=aug_df, train_labels=train_labels, test=False)
    end = process_time() - start
    print(f"acc: {acc} | rt: {end}")
    start = process_time()
    model_type, acc = Decision_tree_classifier(train_df=aug_df, train_labels=train_labels, test=False)
    end = process_time() - start
    print(f"acc: {acc} | rt: {end}")
    start = process_time()
    model_type, acc = Kmeans_classifier(train_df=aug_df, train_labels=train_labels, test=False)
    end = process_time() - start
    print(f"acc: {acc} | rt: {end}")
    start = process_time()
    model_type, acc = KNN_classifier(train_df=aug_df, train_labels=train_labels, test=False)
    end = process_time() - start
    print(f"acc: {acc} | rt: {end}")
    # start = process_time()
    # model_type, acc = SVM_classifier(train_df=aug_df, train_labels=train_labels, test=False)
    # end = process_time() - start
    # print(f"acc: {acc} | rt: {end}")
    # print(acc)
    # print(acc)
    # print(acc)
    # print(acc)
    

def main():
    path = "~/Documents/Syracuse_MS/Summer_2024/IST707/Project/Data/archive (1)/mushroom_overload.csv"
    df = load_data(path=path)
    # print(df)
    aug_df = augment_df(df, 100)
    # print(aug_df)
    train_models(aug_df=aug_df)
    
def test():
    # make_accrt_plot(KNNaccs, model_name="knn")
    path = "~/Documents/Syracuse_MS/Summer_2024/IST707/Project/Data/archive (1)/mushroom_overload.csv"
    df = load_data(path=path)
    aug_df = augment_df(df)
    
    # print(df["cap-diameter"].describe())
    # print(df["stem-width"].describe())
    # print(df["stem-height"].describe())
    # df = remove_cols(df)
    # df = remove_nans(df)
    
    make_stacked_bar_all(aug_df, group_var="cap-shape", normalize=False)
    make_stacked_bar_all(aug_df, group_var="cap-color", normalize=False)
    make_stacked_bar_all(aug_df, group_var="does-bruise-or-bleed", normalize=False)
    make_stacked_bar_all(aug_df, group_var="gill-attachment", normalize=False)
    make_stacked_bar_all(aug_df, group_var="gill-color", normalize=False)
    make_stacked_bar_all(aug_df, group_var="stem-color", normalize=False)
    # col = "season"
    # cat1 = 'a'
    # cat2 = 'u'
    # cat3 = 's'

    # valuescounts = df[col].value_counts()[[ cat1,cat2]]
    # # valuescounts = df["cap-surface"].value_counts()[["t","y"]]

    # print(valuescounts)
    # print(len(df))
    # print(sum(valuescounts))
    # print(sum(valuescounts)/len(df))
    
    
    # valuescounts = df[col].value_counts()[[ cat1]]
    # # valuescounts = df["cap-surface"].value_counts()[["t","y"]]

    # print(valuescounts)
    # print(len(df))
    # print(sum(valuescounts))
    # print(sum(valuescounts)/len(df))
    
    # valuescounts = df[col].value_counts()[[ cat2]]
    # # valuescounts = df["cap-surface"].value_counts()[["t","y"]]

    # print(valuescounts)
    # print(len(df))
    # print(sum(valuescounts))
    # print(sum(valuescounts)/len(df))
    
    # valuescounts = df[col].value_counts()[[ cat3]]
    # # valuescounts = df["cap-surface"].value_counts()[["t","y"]]

    # print(valuescounts)
    # print(len(df))
    # print(sum(valuescounts))
    # print(sum(valuescounts)/len(df))
    
    # print(df[col].value_counts())
    # make_all_hist_sep(df)

    print(df["cap-diameter"].describe())
    print(df["stem-width"].describe())
    print(df["stem-height"].describe())
    # hyperparameter_crawler_v2(aug_df)

def cut_save() -> None:
    path = "~/Documents/Syracuse_MS/Summer_2024/IST707/Project/Data/archive (1)/mushroom_overload.csv"
    df = load_data(path=path)
    aug_df = augment_df(df)
    cut_df = aug_df.iloc[:1000]
    cut_df.to_csv("~/Documents/Syracuse_MS/Summer_2024/IST707/Project/Data/cut_mushroom_dataset.csv", index=False)

if __name__ == '__main__':
    # main()
    # test()
    cut_save()