from kipoi_utils.external.flatten_json import flatten
from bpnet.utils import write_json, dict_prefix_key
from keras.callbacks import EarlyStopping, CSVLogger, TensorBoard
from collections import OrderedDict
import os
import gin
import pandas as pd
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@gin.configurable
class SeqModelTrainer:
    def __init__(self, model, train_dataset, valid_dataset, output_dir,
                 cometml_experiment=None, wandb_run=None):
        """
        Args:
          model: compiled keras.Model
          train: training Dataset (object inheriting from kipoi.data.Dataset)
          valid: validation Dataset (object inheriting from kipoi.data.Dataset)
          output_dir: output directory where to log the training
          cometml_experiment: if not None, append logs to commetml
        """
        # override the model class
        self.seq_model = model
        self.model = self.seq_model.model

        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.cometml_experiment = cometml_experiment
        self.wandb_run = wandb_run
        self.metrics = dict()

        if not isinstance(self.valid_dataset, list):
            # package the validation dataset into a list of validation datasets
            self.valid_dataset = [('valid', self.valid_dataset)]

        # setup the output directory
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.ckp_file = f"{self.output_dir}/model.h5"
        if os.path.exists(self.ckp_file):
            raise ValueError(f"model.h5 already exists in {self.output_dir}")
        self.history_path = f"{self.output_dir}/history.csv"
        self.evaluation_path = f"{self.output_dir}/evaluation.valid.json"

    def train(self,
              batch_size=256,
              epochs=100,
              early_stop_patience=4,
              num_workers=8,
              train_epoch_frac=1.0,
              valid_epoch_frac=1.0,
              train_samples_per_epoch=None,
              validation_samples=None,
              train_batch_sampler=None,
              tensorboard=True):
        """Train the model
        Args:
          batch_size:
          epochs:
          patience: early stopping patience
          num_workers: how many workers to use in parallel
          train_epoch_frac: if smaller than 1, then make the epoch shorter
          valid_epoch_frac: same as train_epoch_frac for the validation dataset
          train_batch_sampler: batch Sampler for training. Useful for say Stratified sampling
          tensorboard: if True, tensorboard output will be added
        """

        if train_batch_sampler is not None:
            train_it = self.train_dataset.batch_train_iter(shuffle=False,
                                                           batch_size=1,
                                                           drop_last=None,
                                                           batch_sampler=train_batch_sampler,
                                                           num_workers=num_workers)
        else:
            train_it = self.train_dataset.batch_train_iter(batch_size=batch_size,
                                                           shuffle=True,
                                                           num_workers=num_workers)
        next(train_it)
        valid_dataset = self.valid_dataset[0][1]  # take the first one
        valid_it = valid_dataset.batch_train_iter(batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=num_workers)
        next(valid_it)

        if tensorboard:
            tb = [TensorBoard(log_dir=self.output_dir)]
        else:
            tb = []

        if self.wandb_run is not None:
            from wandb.keras import WandbCallback
            wcp = [WandbCallback(save_model=False)]  # we save the model using ModelCheckpoint
        else:
            wcp = []

        # train the model
        if len(valid_dataset) == 0:
            raise ValueError("len(self.valid_dataset) == 0")

        if train_samples_per_epoch is None:
            train_steps_per_epoch = max(int(len(self.train_dataset) / batch_size * train_epoch_frac), 1)
        else:
            train_steps_per_epoch = max(int(train_samples_per_epoch / batch_size), 1)

        if validation_samples is None:
            # parametrize with valid_epoch_frac
            validation_steps = max(int(len(valid_dataset) / batch_size * valid_epoch_frac), 1)
        else:
            validation_steps = max(int(validation_samples / batch_size), 1)

        self.model.fit_generator(
            train_it,
            epochs=epochs,
            steps_per_epoch=train_steps_per_epoch,
            validation_data=valid_it,
            validation_steps=validation_steps,
            callbacks=[
                EarlyStopping(
                    patience=early_stop_patience,
                    restore_best_weights=True
                ),
                CSVLogger(self.history_path)
            ] + tb + wcp
        )
        self.model.save(self.ckp_file)

        # log metrics from the best epoch
        try:
            dfh = pd.read_csv(self.history_path)
            m = dict(dfh.iloc[dfh.val_loss.idxmin()])
            if self.cometml_experiment is not None:
                self.cometml_experiment.log_metrics(m, prefix="best-epoch/")
            if self.wandb_run is not None:
                self.wandb_run.summary.update(flatten(dict_prefix_key(m, prefix="best-epoch/"), separator='/'))
        except FileNotFoundError as e:
            logger.warning(e)

    def evaluate(self, metric, batch_size=256, num_workers=8, eval_train=False, eval_skip=[], save=True, **kwargs):
        """Evaluate the model on the validation set
        Args:
          metric: a function accepting (y_true, y_pred) and returning the evaluation metric(s)
          batch_size:
          num_workers:
          eval_train: if True, also compute the evaluation metrics on the training set
          save: save the json file to the output directory
        """
        if len(kwargs) > 0:
            logger.warn(f"Extra kwargs were provided to trainer.evaluate(): {kwargs}")
        # Save the complete model -> HACK
        self.seq_model.save(os.path.join(self.output_dir, 'seq_model.pkl'))

        # contruct a list of dataset to evaluate
        if eval_train:
            eval_datasets = [('train', self.train_dataset)] + self.valid_dataset
        else:
            eval_datasets = self.valid_dataset

        # skip some datasets for evaluation
        try:
            if len(eval_skip) > 0:
                logger.info(f"Using eval_skip: {eval_skip}")
                eval_datasets = [(k, v) for k, v in eval_datasets if k not in eval_skip]
        except Exception:
            logger.warn(f"eval datasets don't contain tuples. Unable to skip them using {eval_skip}")

        metric_res = OrderedDict()
        for d in eval_datasets:
            if len(d) == 2:
                dataset_name, dataset = d
                eval_metric = None  # Ignore the provided metric
            elif len(d) == 3:
                # specialized evaluation metric was passed
                dataset_name, dataset, eval_metric = d
            else:
                raise ValueError("Valid dataset needs to be a list of tuples of 2 or 3 elements"
                                 "(name, dataset) or (name, dataset, metric)")
            logger.info(f"Evaluating dataset: {dataset_name}")
            metric_res[dataset_name] = self.seq_model.evaluate(dataset,
                                                               eval_metric=eval_metric,
                                                               num_workers=num_workers,
                                                               batch_size=batch_size)
        if save:
            write_json(metric_res, self.evaluation_path, indent=2)
            logger.info("Saved metrics to {}".format(self.evaluation_path))

        if self.cometml_experiment is not None:
            self.cometml_experiment.log_metrics(flatten(metric_res, separator='/'), prefix="eval/")

        if self.wandb_run is not None:
            self.wandb_run.summary.update(flatten(dict_prefix_key(metric_res, prefix="eval/"), separator='/'))

        return metric_res
