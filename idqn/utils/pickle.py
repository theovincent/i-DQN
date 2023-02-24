import pickle


def load_pickled_data(path: str):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def save_pickled_data(path: str, object):
    with open(path, "wb") as handle:
        pickle.dump(object, handle)
