import pandas as pd
from sklearn.preprocessing import MinMaxScaler

#reading data
trdf = pd.read_csv("train.csv")
tedf = pd.read_csv("test.csv")



# 1-filling missing values

tedf = tedf.assign(
    Age=tedf["Age"].fillna(tedf["Age"].median()), #median
    Fare=tedf["Fare"].fillna(tedf["Fare"].median()), #median
    Embarked=tedf["Embarked"].fillna(tedf["Embarked"].mode()[0]),#mode
    Cabin=tedf["Cabin"].fillna("Unknown")
)

trdf = trdf.assign(
    Age=trdf["Age"].fillna(trdf["Age"].median()),
    Embarked=trdf["Embarked"].fillna(trdf["Embarked"].mode()[0]),
    Cabin=trdf["Cabin"].fillna("Unknown")
)
#end of section 1


#2- using label encoding for Sex
trdf["Sex"] = trdf["Sex"].map({"male": 0, "female": 1})
tedf["Sex"] = tedf["Sex"].map({"male": 0, "female": 1})

# using one hot encoding for Embarked
trdf = pd.get_dummies(trdf, columns=["Embarked"], drop_first=True)
tedf = pd.get_dummies(tedf, columns=["Embarked"], drop_first=True)
# end of section 2


#3- extracting Name for Title as series by using expand = false
trdf["Title"] = trdf["Name"].str.extract(r' ([A-Za-z]+)\.', expand=False)
tedf["Title"] = tedf["Name"].str.extract(r' ([A-Za-z]+)\.', expand=False)

#extracting the first letter for determining the floor.
# if it's missing it is going to be X.
trdf["Cabin"] = trdf["Cabin"].apply(lambda x: x[0] if x != "Unknown" else "X")
tedf["Cabin"] = tedf["Cabin"].apply(lambda x: x[0] if x != "Unknown" else "X")

#counting the number of the family member.
trdf["FamilySize"] = trdf["SibSp"] + trdf["Parch"] + 1
tedf["FamilySize"] = tedf["SibSp"] + tedf["Parch"] + 1
#end of section 3

#4- deleting unrelated columns
drop_columns = ["PassengerId", "Name", "Ticket"]
trdf.drop(columns=drop_columns, inplace=True)
tedf.drop(columns=drop_columns, inplace=True)
#end of section 4

#5- using MinMaxScaler for normalization.
scaler = MinMaxScaler()
trdf[['Age', 'Fare']] = scaler.fit_transform(trdf[['Age', 'Fare']])
tedf[['Age', 'Fare']] = scaler.transform(tedf[['Age', 'Fare']])
#end of section 5

print(trdf.head())
print(tedf.head())

#6- creating the result
trdf.to_csv("train_result.csv", index=False)
tedf.to_csv("test_result.csv", index=False)

print("Preprocessed files saved successfully!")
