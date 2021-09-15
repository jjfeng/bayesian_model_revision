import logging

import scipy.stats
import numpy as np
import pandas as pd

from constants import SIM_SETTINGS

def make_safe_prob(p, eps=1e-10):
    return np.maximum(eps, np.minimum(1-eps, p))

class FullDatasetMultiPop:
    def __init__(self, full_datasets, proportions, meta_seed: int):
        self.full_datasets = full_datasets
        self.proportions = proportions
        self.num_pops = proportions.size
        self.meta_seed = meta_seed

        # create time map to test batch
        self.time_to_batch_mappings = [{} for i in range(len(self.full_datasets))]
        for pop_idx, dataset in enumerate(self.full_datasets):
            t_raw_idx = 0
            for t_batch, obs_batch in enumerate(dataset.obs_data_list):
                for t in range(t_raw_idx, t_raw_idx + obs_batch.size):
                    self.time_to_batch_mappings[pop_idx][t] = t_batch
                t_raw_idx += obs_batch.size

        self.selected_classes = np.random.choice(np.arange(self.num_pops), p=self.proportions, size=self.size)
        print(self.selected_classes)

    @property
    def true_prob_avail(self):
        return self.full_datasets[0].test_data_list[0].mu is not None

    @property
    def num_batches(self):
        return len(self.full_datasets[0].obs_data_list)

    @property
    def size(self):
        return sum([d.size for d in self.full_datasets[0].obs_data_list])

    def get_train_dat(self):
        np.random.seed(self.meta_seed)
        init_size = self.full_datasets[0].init_train_dat.size
        new_train_datas = []
        for full_dataset, prop in zip(self.full_datasets, self.proportions):
            ntrains = int(init_size * prop)
            selected_idxs = np.random.choice(init_size, size=ntrains, replace=False)
            new_train_datas.append(
                    full_dataset.init_train_dat.subset_idxs(selected_idxs))
        return Dataset.merge(new_train_datas)

    def get_init_recalib_dat(self):
        np.random.seed(self.meta_seed)
        init_size = self.full_datasets[0].init_recalib_dat.size
        new_train_datas = []
        for full_dataset, prop in zip(self.full_datasets, self.proportions):
            ntrains = int(init_size * prop)
            selected_idxs = np.random.choice(init_size, size=ntrains, replace=False)
            new_train_datas.append(
                    full_dataset.init_recalib_dat.subset_idxs(selected_idxs))
        return Dataset.merge(new_train_datas)

    def get_test_dats(self, t_idx: int = None, do_merge: bool = True):
        """
        Grab the test dataset
        """
        if self.full_datasets[0].test_data_list is None:
            return None

        if t_idx is None:
            # return all test data
            test_datas = [
                    Dataset.merge([dataset for dataset in d.test_data_list if dataset is not None], dataset_weights=self.proportions)
                    for d in self.full_datasets]
            return test_datas
        else:
            test_datasets= [
                    d.test_data_list[self.time_to_batch_mappings[pop_idx][t_idx]]
                    for pop_idx, d in enumerate(self.full_datasets)]
            if do_merge:
                test_dataset = Dataset.merge(test_datasets, dataset_weights=self.proportions)
                return test_dataset
            else:
                return test_datasets

    def get_obs(self, t_idx: int, t_idx_end: int):
        all_x = []
        all_y = []
        all_mu = []
        for t in range(t_idx, min(t_idx_end, self.size)):
            selected_class = self.selected_classes[t]
            x, y, mu = self.full_datasets[selected_class].get_obs(t)
            all_x.append(x)
            all_y.append(y)
            all_mu.append(mu)
        return np.vstack(all_x), np.vstack(all_y), np.vstack(all_mu)

    def get_batch_size(self, batch_idx: int):
        return self.full_datasets[0].obs_data_list[batch_idx].size

    def get_timestamp(self, t_idx: int):
        return self.full_datasets[0].get_timestamp(t_idx)

class FullDataset:
    """
    dataset split between training for initial model, training initial recalibration model, and then streaming monitoring data
    """
    def __init__(self, init_train_dat, init_recalib_dat, obs_data_list, test_data_list=None, timestamps=None):
        self.init_train_dat = init_train_dat
        self.init_recalib_dat = init_recalib_dat
        self.obs_data_list = obs_data_list
        self.timestamps = timestamps
        self.test_data_list = test_data_list
        self.merged_dat = Dataset.merge(obs_data_list)

        # create time map to test batch
        self.time_to_batch_mapping = {}
        t_raw_idx = 0
        for t_batch, obs_batch in enumerate(self.obs_data_list):
            for t in range(t_raw_idx, t_raw_idx + obs_batch.size):
                self.time_to_batch_mapping[t] = t_batch
            t_raw_idx += obs_batch.size

    def get_timestamp(self, t_idx: int):
        if self.timestamps is not None:
            return self.timestamps[t_idx]
        else:
            return t_idx

    def get_obs(self, t_idx: int):
        x = self.merged_dat.x[t_idx:t_idx + 1,:]
        y = self.merged_dat.y[t_idx:t_idx + 1,:]
        mu = self.merged_dat.mu[t_idx:t_idx + 1,:] if self.merged_dat.mu is not None else None
        return x, y, mu

class Dataset:
    def __init__(self, x, y, mu: np.ndarray =None, weight: np.ndarray=None):
        self.x = x
        self.y = y
        self.mu = mu
        self.weight = weight

    @property
    def size(self):
        return self.x.shape[0]

    def subset_idxs(self, selected_idxs):
        return Dataset(
                x=self.x[selected_idxs,:],
                y=self.y[selected_idxs,:],
                mu=self.mu[selected_idxs,:] if self.mu is not None else None,
                weight=self.weight[selected_idxs,:] if self.weight is not None else None)

    def subset(self, n, start_n=0):
        assert start_n >= 0
        return Dataset(
                x=self.x[start_n:n,:],
                y=self.y[start_n:n,:],
                mu=self.mu[start_n:n,:] if self.mu is not None else None,
                weight=self.weight[start_n:n,:] if self.weight is not None else None)

    def bootstrap(self):
        idxs = np.random.choice(self.size, self.size)
        return Dataset(
                x=self.x[idxs,:],
                y=self.y[idxs,:],
                mu=self.mu[idxs,:] if self.mu is not None else None,
                weight=self.weight[idxs,:] if self.weight is not None else None)

    @staticmethod
    def merge(datasets, dataset_weights=None):
        """
        @return merged dataset with weights
        """
        has_mu = datasets[-1].mu is not None
        if datasets[-1].weight is not None and dataset_weights is None:
            has_weight = True
            dataset_weights = [1] * len(datasets)
        elif dataset_weights is not None:
            has_weight = True
            for dat in datasets:
                if dat.weight is None:
                    dat.weight = np.ones((dat.size, 1))
        else:
            has_weight = False
        return Dataset(
                x=np.vstack([dat.x for dat in datasets]),
                y=np.vstack([dat.y for dat in datasets]),
                mu=np.vstack([dat.mu for dat in datasets]) if has_mu else None,
                weight=np.vstack([dat.weight * dat_weight for dat, dat_weight in zip(datasets, dataset_weights)]) if has_weight else None)

class DataGenerator:
    train_scale = 2
    test_scale = 2
    def __init__(self, init_beta: np.ndarray, mean_x: np.ndarray, init_perturb: float, meta_seed: int, data_seed: int, simulation: str, subpop: int = None, num_pops: int = None):
        self.init_beta = init_beta
        logging.info("init beta %s", init_beta.ravel())
        self.p = init_beta.size
        self.simulation = simulation
        self.mean_x = mean_x
        self.init_perturb = init_perturb
        self.subpop = subpop
        self.num_pops = 2
        self.meta_seed = meta_seed
        self.data_seed = data_seed

    def generate_data(self, init_train_n, init_recalib_n, test_n: int, change_p: int):
        """
        @param change_p: how many variables to change significnatly
        """
        np.random.seed(self.meta_seed)
        init_train_dat = self.make_data(init_train_n, self.init_beta, scale=self.train_scale)
        init_recalib_dat = self.make_data(init_recalib_n, self.init_beta, scale=self.train_scale)

        change_portion = SIM_SETTINGS[self.simulation]["change_portion"]
        all_batch_meta = SIM_SETTINGS[self.simulation]["all_batch_meta"]
        # Number of periods with constant beta
        T_batches = len(all_batch_meta)
        beta_time_varying = [None] * T_batches

        last_beta = self.init_beta
        last_beta[:change_p] += np.abs(np.random.normal(size=self.init_beta[:change_p].shape)) * self.init_perturb
        logging.info("start beta %s",last_beta.ravel())
        for t, batch_meta in enumerate(all_batch_meta):
            print(batch_meta)
            if batch_meta["copy_old_beta"] is not None:
                new_beta = beta_time_varying[batch_meta["copy_old_beta"]]
            elif batch_meta["do_change"]:
                new_beta = last_beta.copy()
                if "deteriorate" in self.simulation:
                    new_beta[:change_p] = last_beta[:change_p] * (1 - change_portion) + np.abs(np.random.normal(size=(change_p,1))) * change_portion
                else:
                    new_beta[change_p:] += np.random.normal(size=(self.p - change_p,1)) * 0.05 * change_portion
                    new_beta[:change_p] += np.random.normal(size=(change_p,1)) * change_portion
            else:
                new_beta = last_beta
            logging.info("new beta %s",new_beta.ravel())
            beta_time_varying[t] = new_beta
            last_beta = new_beta

        np.random.seed(self.data_seed)
        obs_data_list = [None] * T_batches
        test_data_list = [None] * T_batches
        for t, batch_meta in enumerate(all_batch_meta):
            # make new observations
            obs_data_list[t] = self.make_data(batch_meta["size"], beta_time_varying[t], scale=self.train_scale)
            if batch_meta["copy_old_beta"] is not None:
                # copy old test data if same beta
                test_data_list[t] = test_data_list[batch_meta["copy_old_beta"]]
            else:
                # otherwise generate new test data
                test_data_list[t] = self.make_data(test_n, beta_time_varying[t], scale=self.test_scale)

        full_dat = FullDataset(init_train_dat, init_recalib_dat, obs_data_list, test_data_list)

        return full_dat, [self.init_beta] + beta_time_varying

    def make_data(self, n, beta, scale):
        p = beta.size
        x = np.random.normal(size=(n, p), loc=self.mean_x, scale=scale)

        mu = 1/(1 + np.exp(-(np.matmul(x, beta))))
        y = np.random.binomial(n=1, p=mu, size=(n, 1))

        if self.subpop is not None:
            x = np.hstack([np.ones((x.shape[0], 1)) * (self.subpop - self.num_pops/2), x])

        return Dataset(x, y, mu)

