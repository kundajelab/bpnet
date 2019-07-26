import os
from gin_train.trainers import KerasTrainer
import gin
from collections import OrderedDict
from gin_train.utils import write_json, prefix_dict
from kipoi.external.flatten_json import flatten
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@gin.configurable
class SeqModelTrainer(KerasTrainer):
    def __init__(self, model, train_dataset, valid_dataset, output_dir,
                 cometml_experiment=None, wandb_run=None):
        super().__init__(model, train_dataset, valid_dataset, output_dir,
                         cometml_experiment, wandb_run)
        # override the model class
        self.seq_model = model
        self.model = self.seq_model.model

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
        except:
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
            self.cometml_experiment.log_multiple_metrics(flatten(metric_res, separator='/'), prefix="eval/")

        if self.wandb_run is not None:
            self.wandb_run.summary.update(flatten(prefix_dict(metric_res, prefix="eval/"), separator='/'))

        return metric_res
