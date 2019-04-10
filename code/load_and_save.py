import pickle
import dill

def load_from_pickle(path):
    with open(path, "br") as f:
        return pickle.load(f)
    
def save_lambda_pickle(path, data):
    with open(path, "bw") as f:
        dill.dump(data, f)