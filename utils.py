import pickle


def load_pkl(file_name):
    with open(file_name, "rb") as f:
        obj = pickle.load(f)
    return obj
