"""
create embedding vector to feed into recalibration
"""
import numpy as np
import logging

from dataset import make_safe_prob

class EmbeddingMaker:
    """
    Wrapper for ensemble of locked and/or evolving models
    Does not do interactions or patient subgroups
    """
    def __init__(self, modeler, x_idxs):
        self.modeler = modeler
        self.x_idxs = x_idxs
        self.num_x_idxs = x_idxs.size
        self.centers = 0
        self.scales = 1

        self.num_evolve_models = len(self.modeler.evolve_idxs)
        self.locked_embed_idxs = np.array(
                self.modeler.locked_idxs
                + list(range(1 + self.num_evolve_models, 1 + self.num_evolve_models + self.num_x_idxs))
        )
        self.full_embed_dim = self.modeler.num_models + self.num_x_idxs

    def remap_model_locked_to_full_params(self, locked_params: np.ndarray):
        """
        @param locked_params: parameters for the locked recalibrator
        @return remapped version of the parameters from the locked recalibrator (based on which variables generated by the locked versus evolving models)
        """
        full_params = np.zeros(1 + self.full_embed_dim)
        # Copy intercept
        full_params[0] = locked_params[0]
        # Copy coefficient for locked model
        full_params[self.locked_embed_idxs + 1] = locked_params[1:]
        return full_params

    def refit_pred_probs(self, newx, refitting_labels=None):
        # If the model refits with accumulating labels, we need to actually run refitting
        pred_probs_raw = []
        for i in range(0,newx.shape[0], self.modeler.refit_freq):
            pred_probs_raw.append(self.modeler.predict_prob(newx[i:i+self.modeler.refit_freq]))
            self.modeler.update(newx[i:i+self.modeler.refit_freq], refitting_labels[i:i+self.modeler.refit_freq], is_init=True)
        return np.vstack(pred_probs_raw)

    def _embed(self, newx, refitting_labels=None):
        """
        @return tuple with:
             embedding from refitting model
             embedding from locked model
             predicted probabilities from the refitted model
        """
        if refitting_labels is None or self.modeler.refit_freq is None:
            # Static modeler
            pred_probs_raw = self.modeler.predict_prob(newx)
        else:
            # special embedding with refitting
            pred_probs_raw = self.refit_pred_probs(newx, refitting_labels)

        pred_probs = make_safe_prob(pred_probs_raw)
        pred_logit = np.log(pred_probs/(1 - pred_probs))
        locked_logits = pred_logit[:,0:1]

        model_pred_probs = pred_probs_raw[:,-1].reshape((-1,1))

        if self.x_idxs.size > 0:
            linear_x = newx[:,self.x_idxs]
            return np.hstack([pred_logit, linear_x]), np.hstack([locked_logits, linear_x]), model_pred_probs
        else:
            return pred_logit, locked_logits, model_pred_probs

    def embed(self, newx, refitting_labels=None):
        evolve_embed, locked_embed, model_pred_probs = self._embed(newx, refitting_labels=refitting_labels)
        return (evolve_embed - self.centers)/self.scales, (locked_embed - self.locked_centers)/self.locked_scales, model_pred_probs

    def initialize(self, newx):
        """
        Run this the first time on the initial recalibration dataset (only on the variables) to learn
        how to center and scale embeddings.

        @return tuple with:
             embedding from refitting model
             embedding from locked model
             predicted probabilities from the refitted model
        """
        raw_evolve_embed, raw_locked_embed, _ = self._embed(newx)
        self.locked_centers = raw_locked_embed.mean(axis=0, keepdims=True)
        self.locked_scales = np.sqrt(raw_locked_embed.var(axis=0, keepdims=True))
        self.centers = raw_evolve_embed.mean(axis=0, keepdims=True)
        self.scales = np.sqrt(raw_evolve_embed.var(axis=0, keepdims=True))

        evolve_embed, locked_embed, model_pred_probs = self.embed(newx)
        return evolve_embed, locked_embed, model_pred_probs

class SubgroupsEmbeddingMaker(EmbeddingMaker):
    """
    Create embedding if you have two patient subgroups
    Embedding format: y ~ pred_score * is_group0 + pred_score * is_group1
    """
    def __init__(self, modeler, group_idxs, num_groups: int = 2):
        assert num_groups == 2
        # This is not for evolving models
        assert len(modeler.evolve_idxs) == 0

        self.modeler = modeler
        self.num_groups = num_groups
        self.group_idxs = group_idxs
        self.centers = 0
        self.scales = 1

        self.locked_embed_idxs = np.arange(self.num_groups)
        self.full_embed_dim = self.locked_embed_idxs

    def remap_model_locked_to_full_params(self, locked_params: np.ndarray):
        """
        @param locked_params: parameters for the locked recalibrator
        @return remapped version of the parameters from the locked recalibrator (based on which variables generated by the locked versus evolving models)
        """
        return locked_params

    def _embed(self, newx, refitting_labels=None):
        """
        @return tuple with:
             embedding from refitting model
             embedding from locked model
             predicted probabilities from the refitted model
        """
        if refitting_labels is None or self.modeler.refit_freq is None:
            # Static modeler
            pred_probs_raw = self.modeler.predict_prob(newx)
        else:
            # special embedding with refitting
            pred_probs_raw = self.refit_pred_probs(newx, refitting_labels)

        pred_probs = make_safe_prob(pred_probs_raw)
        pred_logit = np.log(pred_probs/(1 - pred_probs))
        model_pred_probs = pred_probs_raw[:,-1].reshape((-1,1))

        group_x = newx[:,self.group_idxs]
        assert np.max(group_x) < self.num_groups
        # Currently, the interacitons only work if there is only one predicted probability
        assert group_x.shape == pred_logit.shape
        embedding = np.hstack([pred_logit * (1 - group_x), pred_logit * group_x])
        return embedding, embedding, model_pred_probs

class EvolveOnlyEmbeddingMaker(EmbeddingMaker):
    """
    Creates embeddings for evolving only models
    """
    def __init__(self, modeler, group_idxs, num_groups: int = 2):
        assert num_groups == 2
        # This is not for evolving models
        assert len(modeler.evolve_idxs) == 0

        self.modeler = modeler
        self.num_groups = num_groups
        self.group_idxs = group_idxs
        self.centers = 0
        self.scales = 1

        self.num_evolve_models = len(self.modeler.evolve_idxs)
        self.locked_embed_idxs = np.arange(self.num_evolve_models + self.num_x_idxs)
        self.full_embed_dim = self.locked_embed_idxs

    def remap_model_locked_to_full_params(self, locked_params: np.ndarray):
        """
        @param locked_params: parameters for the locked recalibrator
        @return remapped version of the parameters from the locked recalibrator (based on which variables generated by the locked versus evolving models)
        """
        return locked_params
