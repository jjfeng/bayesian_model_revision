from typing import List
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from dataset import Dataset

class LockedModeler:
    def __init__(self, dat: Dataset, n_estimators: int=200, max_depth: int = 3):
        self.dat = dat
        self.curr_model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=.04, max_depth=max_depth, random_state=0)
        self._fit_model()
        self.refit_freq = None

    def _fit_model(self):
        self.curr_model.fit(self.dat.x, self.dat.y.flatten())

    def predict_prob_single(self, x):
        return self.curr_model.predict_proba(x)[:,1].reshape((-1,1))

    def predict_prob(self, x):
        return self.curr_model.predict_proba(x)[:,1].reshape((-1,1))

    def update(self, x, y, is_init=False):
        """
        @return whether or not the underlying model changed
        """
        # Do nothing
        return False

    @property
    def num_models(self):
        return len(self.locked_idxs + self.evolve_idxs)

    @property
    def locked_idxs(self):
        return [0]

    @property
    def evolve_idxs(self):
        return []

class BoxedModeler(LockedModeler):
    """
    This modeler uses only the most recent data and if specified, will randomly purge the training data.
    """
    def __init__(self, dat: Dataset, n_estimators: int=200, max_depth: int = 3, refit_freq: int = 1, max_trains: List[int] = [100], switch_time: int=None):
        """
        @param switch_time: when to switch number of training obs
        """
        self.refit_freq = refit_freq
        self.dat = dat
        self.max_trains = max_trains
        self.max_train_idx = 0
        self.curr_model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=.04, max_depth=max_depth, random_state=1)
        shuffle_dat = dat.subset_idxs(np.random.choice(dat.size, dat.size, replace=False))
        self.dat = shuffle_dat.subset(dat.size, dat.size//10)
        self._fit_model()
        self.dat = dat
        self.track_idx = 0
        self.num_update = 0
        self.switch_time = switch_time

    def update(self, x, y, is_init=False):
        self.num_update += 1
        if self.num_update == self.switch_time:
            print("SWITCHY TIME")
            self.max_train_idx = min(self.max_train_idx + 1, len(self.max_trains) - 1)
        self.dat = Dataset.merge([self.dat, Dataset(x, y)])

        orig_batch_num = self.track_idx // self.refit_freq
        self.track_idx += x.shape[0]
        new_batch_num = self.track_idx // self.refit_freq
        if new_batch_num > orig_batch_num:
            self.dat = self.dat.subset(self.dat.size, start_n=max(0, self.dat.size - self.max_train))
            print("REFIT data", self.dat.size)
            self._fit_model()
            return True
        else:
            return False

    @property
    def max_train(self):
        return self.max_trains[self.max_train_idx]

    @property
    def locked_idxs(self):
        return []

    @property
    def evolve_idxs(self):
        return [0]

class CumulativeModeler(LockedModeler):
    def __init__(self, dat: Dataset, n_estimators: int=200, max_depth: int = 3, refit_freq: int = 1):
        self.refit_freq = refit_freq
        self.curr_model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=.04, max_depth=max_depth, random_state=1)
        shuffle_dat = dat.subset_idxs(np.random.choice(dat.size, dat.size, replace=False))
        self.dat = shuffle_dat.subset(dat.size, dat.size//10)
        self._fit_model()
        self.dat = dat

    def update(self, x, y, is_init=False):
        orig_batch_num =  self.dat.size // self.refit_freq
        self.dat = Dataset.merge([self.dat, Dataset(x, y)])
        new_batch_num = self.dat.size//self.refit_freq
        if new_batch_num > orig_batch_num:
            self._fit_model()
            return True
        else:
            return False

    @property
    def locked_idxs(self):
        return []

    @property
    def evolve_idxs(self):
        return [0]

class ComboModeler(LockedModeler):
    def __init__(self, modelers, refit_freq):
        assert len(modelers) == 2
        self.modelers = modelers
        self.refit_freq = refit_freq

    def predict_prob_single(self, x):
        """
        @return prob from the last model
        """
        return self.modelers[-1].predict_prob_single(x)

    def predict_prob(self, x):
        """
        @return prob from all the models
        """
        all_probs = np.hstack([m.predict_prob(x) for m in self.modelers])
        return all_probs

    def update(self, x, y, is_init=False):
        did_update = False
        for m in self.modelers:
            did_m_update = m.update(x, y, is_init)
            did_update = did_update or did_m_update
        return did_update

    @property
    def locked_idxs(self):
        return [idx for idx, m in enumerate(self.modelers) if m.locked_idxs]

    @property
    def evolve_idxs(self):
        return [idx for idx, m in enumerate(self.modelers) if m.evolve_idxs]

