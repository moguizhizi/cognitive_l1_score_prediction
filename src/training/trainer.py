from src.utils.logger import get_logger

logger = get_logger(__name__)


class Trainer:

    def __init__(self, model):
        self.model = model

    def fit(self, X_train, y_train, X_val=None, y_val=None, sample_weight=None):

        logger.info("Start model training")

        self.model.fit(X_train, y_train, sample_weight=sample_weight)

        logger.info("Training finished")

        if X_val is not None:

            logger.info("Running validation prediction")

            val_pred = self.model.predict(X_val)

            return val_pred

        return None
