import keras
import numpy as np
import sklearn.metrics as sklm


class Metrics(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.confusion = []
        self.precision = []
        self.recall = []
        self.f1score = []
        self.kappa = []
        self.auc = []

    def on_epoch_end(self, epoch, logs={}):
        score = np.asarray(self.model.predict(self.validation_data[:-3]))
        predict = np.argmax(score, axis=1)
        targ = np.argmax(self.validation_data[-3], axis=1)

        self.auc.append(sklm.roc_auc_score(targ, score[:,1]))
        self.confusion.append(sklm.confusion_matrix(targ, predict))
        self.precision.append(sklm.precision_score(targ, predict))
        self.recall.append(sklm.recall_score(targ, predict))
        self.f1score.append(sklm.f1_score(targ, predict))
        self.kappa.append(sklm.cohen_kappa_score(targ, predict))

        print('Kappa = %.4f, F1S = %.4f, AUC = %.4f' % (self.kappa[-1], self.f1score[-1], self.auc[-1]))

        return


def convert_keras_metric_to_dict(m_obj):
    attributes = [a for a in dir(m_obj) if a not in dir(keras.callbacks.Callback) and a != "model"]
    return {attr: m_obj.__getattribute__(attr) for attr in attributes}
