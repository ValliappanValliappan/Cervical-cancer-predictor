from sklearn import svm
import pandas
classifier=svm.SVC()
columns=["Age","Numberofsexualpartners", "Firstsexualintercourse",
         "Numofpregnancies","Smokes", "Smokesyears","Smokespacksyear", "HormonalContraceptives",
         "HormonalContraceptivesyears", "IUD", "IUDyears", "STDs", "STDsnumber", "STDscondylomatosis",
         "STDscervicalcondylomatosis", "STDsvaginalcondylomatosis","STDsvulvoperinealcondylomatosis",
         "STDssyphilis", "STDspelvicinflammatorydisease","STDsgenitalherpes","STDsmolluscumcontagiosum",
         "STDsAIDS", "STDsHIV","STDsHepatitisB","STDsHPV","STDsNumberofdiagnosis",
         "DxCancer","DxCIN","DxHPV","Dx"]
training_data=pandas.read_csv("/Users/valli/Desktop/hi.csv", names=columns)#Look at scratch_3.py for why its hi.csv
target=pandas.read_csv("/Users/valli/Downloads/target.csv", names=["Label"])
y=[]
for i in target.values:
    for j in i:
        y.append(j)
X=training_data.values
classifier.fit(X=X, y=y)
print(classifier.score(X,y))