import numpy as np
import pickle

class ifp():
    def __init__(self,data):
        self.data = data

    def load_model(self):
        with open(r"artifacts/model.pkl","rb") as file:
            self.model=pickle.load(file)

    def predict(self):
        self.load_model()


        Sepal_length = float(self.data["Sepal_length"])
        Sepal_width = float(self.data["Sepal_width"])
        petal_length = float(self.data["petal_length"])
        petal_width = float(self.data["petal_width"])

        array = np.array([Sepal_length,Sepal_width,petal_length,petal_width],ndmin=2)
        print(array)
        print("*"*100)

        res = self.model.predict(array)[0]
        print(res)

        return res
if __name__ == "__main__":

    data={
        "Sepal_length" : 5.1,
        "Sepal_width" : 3.8,
        "petal_length" : 1.5,
        "petal_width" : 0.3
    }




    ifp_obj = ifp(data)

    ifp_obj.predict()

