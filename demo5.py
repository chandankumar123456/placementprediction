import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


#Step 1
#Loading the dataset:

df= pd.read_csv("dataset1.csv")


#Step 2
#Data preproccessing

le=LabelEncoder()
stream=le.fit_transform(df['Stream'])
df["Stream"]=stream
x=df.pop("Stream")
df.insert(2,"Stream",x)


x=le.fit_transform(df["Gender"])
df.drop("Gender",axis=1,inplace=True)
df.insert(1,"Gender",x)

#Step 3
#Building our model

x_train,x_test,y_train,y_test=train_test_split(df[['Age','Gender','Stream','Internships','CGPA','HistoryOfBacklogs']],df.PlacedOrNot,test_size=0.2)
model=LogisticRegression()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)

#Step 4
#Building Streamlit app

def fun():
	st.header("Placement Prediction Project")
	st.info("Enter all the details properly")
	age=st.number_input("Enter your age")
	gen=st.radio("Enter your gender",["Male", "Female"])
	Stream=st.radio("Enter your stream",["CSE","IT","ECE","MECH","CIVIL"])
	interns=st.number_input("Enter how many internships have you done:")
	cgpa=st.number_input("Enter your CGPA:")
	back=st.number_input("Enter HistoryOfBacklogs")
	if gen == "Male":
		gen = 1
	else:
		gen = 0

	if Stream=="CSE":
		Stream=1
	elif Stream=="IT":
		Stream=4
	elif Stream=="ECE":
		Stream=3
	elif Stream=="MECH":
		Stream=5
	else:
		Stream=2
	li=[age,gen,Stream,interns,cgpa,back]
	x=st.button("SUBMIT")
	if x:
		output=model.predict([li])
		if output == 1:
			st.success("Yes you got the placement")
		else:
			st.error("No You did not get the placement")


fun()
