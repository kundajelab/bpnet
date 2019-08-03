"""Other gin configurables
"""
import os
import gin
import modisco
import modisco.tfmodisco_workflow.workflow
from gin import config
import keras

# keras.optimizers
config.external_configurable(keras.optimizers.Adam, module='keras.optimizers')
config.external_configurable(keras.optimizers.RMSprop, module='keras.optimizers')
config.external_configurable(keras.optimizers.Adagrad, module='keras.optimizers')
config.external_configurable(keras.optimizers.Adadelta, module='keras.optimizers')
config.external_configurable(keras.optimizers.Adamax, module='keras.optimizers')
config.external_configurable(keras.optimizers.Nadam, module='keras.optimizers')
config.external_configurable(keras.optimizers.SGD, module='keras.optimizers')


# modisco
config.external_configurable(modisco.tfmodisco_workflow.workflow.TfModiscoWorkflow)
config.external_configurable(modisco.tfmodisco_workflow.seqlets_to_patterns.TfModiscoSeqletsToPatternsFactory)


@gin.configurable
def report_template(name, raise_error=True):
    """Evaluation report template found in ../templates/
    """
    import inspect
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    this_dir = os.path.dirname(os.path.abspath(filename))
    template_file = os.path.join(this_dir, 'templates/', name)
    if not os.path.exists(template_file):
        if raise_error:
            # list all file
            available_template_files = [f for f in os.listdir(os.path.dirname(template_file))
                                        if f.endswith('.ipynb')]
            template_str = "\n".join(available_template_files)
            raise FileNotFoundError(
                f"Template {name} doesn't exist. Available templates:\n{template_str}"
            )
        else:
            return None
    return template_file


# alias
eval_report_template = report_template

