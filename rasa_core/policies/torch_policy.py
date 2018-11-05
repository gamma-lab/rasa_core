"""A RNN Policy Implemented in PyTorch"""
from rasa_core import utils
from rasa_core.policies.policy import Policy
from rasa_core.featurizers import TrackerFeaturizer
from sklearn.metrics import accuracy_score

import io
import json
import logging
import os
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class TorchPolicy(Policy):
    SUPPORTS_ONLINE_TRAINING = True
    defaults = {
        # Default NN params
        "rnn_size": 64,
        "layers": 2
    }
    def __init__(self,
                 featurizer=None,  # type: Optional[TrackerFeaturizer]
                 model=None,  # type: Optional[torch.nn.RNN]
                 current_epoch=0  # type: int
                 ):

        super(TorchPolicy, self).__init__(featurizer)

        self.model = model
        self.hidden_size = defaults['rnn_size']
        self.layers = defaults['layers']
        self.current_epoch = current_epoch

    def model_architecture(self,
                           input_size,  # type: Tuple[int, int]
                           output_size  # type Tuple[int, int]
                           ):

        # assumes output_shape is size (num examples, num features)
        rnn = RNN(input_size,
                  self.hidden_size,
                  output_size,
                  self.layers,
                  dropout=0.2)
        return rnn

    def train(self,
              training_trackers,  # type: List[DialogueStateTracker]
              domain,  # type: Domain
              **kwargs  # type: Any
              ):

        training_data = self.featurize_for_training(training_trackers, domain)
        shuffled_X, shuffled_y = training_data.shuffled_X_y()
        batch_size, seq_len, input_size = shuffled_X.shape
        batch_size, output_size = shuffled_y.shape

        if kwargs.get('rnn_size') is not None:
            logger.debug("Parameter rnn_size updated with "
                         "{}".format(kwargs.get("rnn_size")))
            self.hidden_size = kwargs.get('rnn_size')

        if self.model is None:
            self.model = self.model_architecture(input_size,
                                                 output_size
                                                 )
            self.input_size = input_size
            self.output_size = output_size

        val_split = kwargs.get("validation_split", 0.1)
        train_size = int(shuffled_X.shape[0] * (1-val_split))

        # convert to float since model requires float
        train_X = torch.tensor(shuffled_X[:train_size]).type(torch.float)
        train_y = torch.tensor(shuffled_y[:train_size]).type(torch.float)
        val_X = torch.tensor(shuffled_X[train_size:]).type(torch.float)
        val_y = torch.tensor(shuffled_y[train_size:]).type(torch.float)

        del shuffled_X, shuffled_y

        logger.info("Fitting model with {} total samples and validation split "
                    "of {}".format(training_data.num_examples(), val_split))

        # hyperparams and optimizer
        epochs = kwargs.get("epochs", 30)
        self.current_epoch += epochs
        batch_size = kwargs.get("batch_size", 16)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        for ep in range(epochs):
            permutation = torch.randperm(train_X.shape[0])
            # for printing acc.
            print_loss = 0
            train_acc = 0
            print_val_loss = 0
            val_acc = 0
            train_it = 0
            val_it = 0

            # train batches
            for i in range(0, train_X.shape[0], batch_size):
                optimizer.zero_grad()
                idx = permutation[i:i + batch_size]
                out = self.model(train_X[idx])[:, -1, :]
                loss = -torch.mean(torch.sum(train_y[idx] * out, dim=1))
                pred_y = torch.argmax(out, dim=1)
                train_acc += accuracy_score(torch.argmax(train_y[idx], dim=1),
                                            pred_y)
                print_loss += loss.item()
                train_it += 1
                loss.backward()
                optimizer.step()

            for i in range(0, val_X.shape[0], batch_size):
                val_it += 1
                with torch.no_grad():
                    out = self.model(val_X[i:i + batch_size])[:, -1, :]
                    loss = -torch.mean(torch.sum(val_y[i:i + batch_size]
                                                 * out, dim=1))
                    pred_y = torch.argmax(out, dim=1)
                    val_acc += accuracy_score(
                        torch.argmax(val_y[i:i + batch_size], dim=1),
                        pred_y)

                    print_val_loss += loss.item()

            logger.info("Epoch {} | Train loss/acc: {} / {} | "
                        "Val loss/acc {} / {}"
                        .format(ep + 1,
                                round(print_loss, 3),
                                round(train_acc / train_it, 3),
                                round(print_val_loss, 3),
                                round(val_acc / val_it, 3)))

    def continue_training(self, training_trackers, domain, **kwargs):
        logger.info("Continuing training for model")
        batch_size = kwargs.get("batch_size", 16)
        epochs = kwargs.get("epochs", 10)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        for ep in range(epochs):
            training_data = self._training_data_for_continue_training(
                batch_size, training_trackers, domain)
            X, y = self._convert_to_tensors(training_data)
            optimizer.zero_grad()
            out = self.model(X)[:, -1, :]
            loss = -torch.mean(torch.sum(y * out, dim=1))

            loss.backward()
            optimizer.step()
            self.current_epoch += 1
            logger.info("Epoch {} | Train Loss: {}"
                        .format(self.current_epoch + 1, loss.item()))

    def predict_action_probabilities(self, tracker, domain):
        # type: (DialogueStateTracker, Domain) -> List[float]

        # noinspection PyPep8Naming
        X = self.featurizer.create_X([tracker], domain)
        X = torch.tensor(X).type(torch.float)
        with torch.no_grad():
            y_pred = self.model(X)

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
                return cls(featurizer=featurizer,
                           model=model
                           )
            else:
                return cls(featurizer=featurizer)
        else:
            raise Exception("path {} does not exist".format(path))

    def _convert_to_tensors(self, data):
        """Returns X, y from data as torch float tensors"""
        X = torch.tensor(data.X).type(torch.float)
        y = torch.tensor(data.y).type(torch.float)
        return X, y


class RNN(nn.Module):
    def __init__(self,
                 input_size,  # type: int
                 hidden_size,  # type: int
                 output_size,  # type: int
                 layers,  # type: int
                 dropout  # type: float
                 ):
        super(RNN, self).__init__()
        self.rnn = nn.GRU(input_size,
                          hidden_size,
                          layers,
                          batch_first=True,
                          dropout=dropout
                          )
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        # forward pass
        out, hidden = self.rnn(inputs)
        out = F.log_softmax(self.out(out), dim=2)
        return out
