class Trainer:

    def __init__(self, model):
        self.model = model

    def fit(self, X_train, y_train, X_val=None, y_val=None):

        self.model.fit(X_train, y_train)

        if X_val is not None:
            val_pred = self.model.predict(X_val)
            return val_pred

        return None
