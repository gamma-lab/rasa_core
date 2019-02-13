"""A RNN Policy Implemented in PyTorch"""
from rasa_core import utils
from rasa_core.policies.policy import Policy
from rasa_core.featurizers import (
    MaxHistoryTrackerFeaturizer, BinarySingleStateFeaturizer, TrackerFeaturizer)
from sklearn.metrics import accuracy_score

import io
import json
import copy
import logging
import os
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D
import numpy as np


logger = logging.getLogger(__name__)


class TorchPolicy(Policy):
    SUPPORTS_ONLINE_TRAINING = True

    allowed_params = ("epochs", "batch_size", "validation_split", "learning_rate", "device")

    defaults = {
        "epochs": 50,
        "batch_size": 8,
        "validation_split": 0.1,
        "learning_rate": 1e-3,
        "device": "cpu",
    }

    @staticmethod
    def _standard_featurizer(max_history=None):
        return MaxHistoryTrackerFeaturizer(
            BinarySingleStateFeaturizer(), max_history=max_history)

    def _load_params(self, **kwargs):
        params = copy.deepcopy(self.defaults)
        params.update(kwargs)
        for p in params.keys():
            if not p in self.allowed_params:
                logger.warn("Unknown policy parameter: %s", p)
            setattr(self, p, params.get(p))

    def __init__(
            self,
            featurizer=None,
            model=None,
            graph=None,
            session=None,
            current_epoch=0,
            **kwargs
    ):
        if not featurizer:
            featurizer = self._standard_featurizer(kwargs.get("max_history"))
        super(TorchPolicy, self).__init__(featurizer)
        self.model = model
        self.graph = graph
        self.session = session
        self.current_epoch = current_epoch
        self._load_params(**kwargs)

    def model_architecture(self, input_size, output_size, multi_output):
        config = {
            "input_size": input_size,
            "output_size": output_size,
            "multi_output": multi_output}
        model = PolicyModel(config)
        return model.to(self.device)

    def train_test_split(self, x, ratio):
        # assumes batch first
        offset = int(x.shape[0] * ratio)
        train, test = x[offset:], x[:offset]
        return train, test

    def featurize_for_training(
            self,
            training_trackers,
            domain,
            **kwargs
    ):
        feature = super(TorchPolicy, self).featurize_for_training(
            training_trackers, domain, **kwargs)
        feature.X = torch.tensor(feature.X).float()
        feature.y = torch.tensor(feature.y).float()
        return feature

    def train(
            self,
            training_trackers,
            domain,
            **kwargs
    ):
        training_data = self.featurize_for_training(training_trackers, domain, **kwargs)
        feature_x, feature_y = training_data.shuffled_X_y()

        x_seq_len = None
        y_seq_len = None
        multi_output = len(feature_y.shape) > 2

        if multi_output:
            x_seq_len = feature_x[1]
            y_seq_len = feature_y[1]

        if self.model is None:
            self.model = self.model_architecture(input_size=feature_x.shape[-1],
                output_size=feature_y.shape[-1], multi_output=multi_output)
            self.model.init_weight()

        optim = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        logger.info("Fitting model with %s total samples and %s validation split",
                    training_data.num_examples(), self.validation_split)

        self.continue_training(training_trackers, domain, **kwargs)
        for ep in range(self.epochs):
            feature_x, feature_y = training_data.shuffled_X_y()  # can shuffle existing vector too
            train_x, val_x = self.train_test_split(feature_x, self.validation_split)
            train_y, val_y = self.train_test_split(feature_y, self.validation_split)

            train_iterator = D.DataLoader(
                ZipDataset(train_x, train_y),
                batch_size=self.batch_size,
                collate_fn=mini_batcher)

            val_iterator = D.DataLoader(
                ZipDataset(val_x, val_y),
                batch_size=self.batch_size,
                collate_fn=mini_batcher)

            train_len, val_len = len(train_iterator), len(val_iterator)
            train_metrics = MetricCollection()
            val_metrics = MetricCollection()

            for batch_x, batch_y in train_iterator:
                optim.zero_grad()
                predicted, _ = self.model(batch_x)
                real_labels = torch.argmax(batch_y, dim=-1)
                loss = calculate_loss(predicted, real_labels, multi_output)
                loss.backward()
                optim.step()
                train_metrics += calculate_metrics(predicted, real_labels, loss)

            with torch.no_grad():
                for batch_x, batch_y in val_iterator:
                    predicted, _ = self.model(batch_x)
                    real_labels = torch.argmax(batch_y, dim=-1)
                    loss = calculate_loss(predicted, real_labels, multi_output)
                    val_metrics += calculate_metrics(predicted, real_labels, loss)

            logger.info("Epoch %s | TRAIN %s | VAL %s", ep + 1,
                train_metrics.div(train_len), val_metrics.div(val_len))

        # done..
        self.current_epoch = self.epochs

    def continue_training(self, training_trackers, domain, **kwargs):
        logger.info("Continuing training for model")
        batch_size = kwargs.get("batch_size", 8)
        epochs = kwargs.get("epochs", 10)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        for ep in range(epochs):
            training_data = self._training_data_for_continue_training(
                batch_size, training_trackers, domain)
            iterator = D.DataLoader(
                ZipDataset(training_data.X, training_data.y),
                batch_size=batch_size,
                collate_fn=mini_batcher)
            multi_output = len(training_data.y.shape) > 2
            metrics = MetricCollection()

            for X, y in iterator:
                optimizer.zero_grad()
                real_labels = torch.argmax(training_data.y, dim=-1)
                predicted, _ = self.model(training_data.X)
                loss = calculate_loss(predicted, real_labels, multi_output)
                loss.backward()
                optimizer.step()
                metrics += calculate_metrics(predicted, real_labels, loss)

            self.current_epoch += 1
            logger.info("Epoch %s | %s", self.current_epoch, metrics.div(batch_size))

    def predict_action_probabilities(
            self,
            tracker,
            domain
    ):
        X = self.featurizer.create_X([tracker], domain)
        X = torch.tensor(X).float()
        with torch.no_grad():
            y_pred, _ = self.model(X)
            y_pred = np.exp(y_pred)
        if len(y_pred.shape) == 2:
            return y_pred[-1].tolist()
        elif len(y_pred.shape) == 3:
            return y_pred[0, -1].tolist()

    def persist(self, path):
        # tupe: (Text) -> None
        if self.model:
            self.featurizer.persist(path)
            meta = {"model": "torch_model.h5",
                    "epochs": self.current_epoch}
            config_file = os.path.join(path, "torch_policy.json")
            utils.dump_obj_as_json_to_file(config_file, meta)

            model_file = os.path.join(path, meta["model"])
            utils.create_dir_for_file(model_file)
            torch.save(self.model, model_file)
        else:
            warnings.warn("Persist called without a trained model present. "
                          "Nothing to persist.")

    @classmethod
    def load(cls, path):
        # type: (Text) -> TorchPolicy
        if os.path.exists(path):
            featurizer = TrackerFeaturizer.load(path)
            meta_path = os.path.join(path, "torch_policy.json")
            if os.path.isfile(meta_path):
                with io.open(meta_path) as f:
                    meta = json.loads(f.read())
                model_file = os.path.join(path, meta["model"])
                model = torch.load(model_file)
                return cls(featurizer=featurizer, model=model)
            else:
                return cls(featurizer=featurizer)
        else:
            raise Exception("path {} does not exist".format(path))


def mini_batcher(xs):
    return (
        torch.stack([x[0] for x in xs]),
        torch.stack([x[1] for x in xs]))

def calculate_loss(
        y_pred_likelihood,  # (B,S,F)
        y_real_labels,    # (B,S,F) or (B,F)
        multi_output=False,
):
    if multi_output:
        y_pred_likelihood = y_pred_likelihood.permute(0,2,1)  # B,F,S
    loss = F.nll_loss(y_pred_likelihood, y_real_labels)
    return loss


def calculate_metrics(
        y_pred_likelihood,
        y_real_labels,
        loss,
):
    y_pred_labels = torch.argmax(y_pred_likelihood, dim=-1)
    return {
        "acc": accuracy_score(y_real_labels.flatten(), y_pred_labels.flatten()),
        "loss": loss.item(),
    }


class ZipDataset(D.Dataset):
    def __init__(self, *ds):
        self.ds = ds

    def __len__(self):
        return min(len(d) for d in self.ds)

    def __getitem__(self, idx):
        return tuple(d[idx] for d in self.ds)


class MetricCollection(dict):

    def __add__(self, x):
        new = copy.deepcopy(self)
        for k, v in x.items():
            new[k] = new.get(k, 0) + v
        return new

    def div(self, x):
        return MetricCollection((k, v/x) for k,v in self.items())

    def __repr__(self):
        return " | ".join("{}: {:.3f}".format(k, v) for k,v in self.items())


class PolicyModel(nn.Module):

    default_config = {
        "hidden_size": 32,
        "layers": 1,
        "directions": 1,
        "dropout": 0.2,
        "multi_output": False,
    }

    def __init__(self, config):
        super(PolicyModel, self).__init__()
        self.config = copy.deepcopy(self.default_config)
        self.config.update(config)

        self.rnn = nn.LSTM(self.config["input_size"], self.config["hidden_size"],
            self.config["layers"], dropout=self.config["dropout"], batch_first=True)
        self.lin = nn.Linear(self.config["hidden_size"], self.config["output_size"])

    def forward(self, inputs, hidden=None):
        out, hidden = self.rnn(inputs, hidden)
        out = self.lin(out)
        if not self.config["multi_output"]:
            out = out[:,-1,:]
        out = F.log_softmax(out, dim=-1)
        return out, (hidden[0].detach(), hidden[1].detach())

    def init_hidden(self, batch):
        c = self.config
        return (
            torch.zeros(c["layers"] * c["directions"], batch, c["hidden_size"]),
            torch.zeros(c["layers"] * c["directions"], batch, c["hidden_size"]))

    def init_weight(self):
        for n, p in self.named_parameters():
            if 'weight' in n:
                torch.nn.init.xavier_normal_(p)
            elif 'bias' in n:
                torch.nn.init.zeros_(p)
