import pickle
def result(n,p,k,temp,hum,ph,rain):
    #SVM-Support Vector Machine , Decision Tree, KNN K-Nearest Neighbor
    model=pickle.load(open('ensemble_classifier.pkl','rb'))
    res= model.predict([[n,p,k,temp,hum,ph,rain]])
    print(f"\n\n{res}\n\n")
    return res
    