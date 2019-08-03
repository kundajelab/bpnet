import gin


class DummyModiscoResult:
    def save_hdf5(self, grp):
        pass


@gin.configurable
class DummyModiscoWorkflow:
    """Dummy configurable to test
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        print(f"Recieved args: {args}")
        print(f"Recieved kwargs: {kwargs}")

    def __call__(self, *args, **kwargs):
        req_kwargs = ['task_names',
                      'contrib_scores',
                      'hypothetical_contribs',
                      'one_hot',
                      'null_per_pos_scores']
        for kw in req_kwargs:
            assert kw in kwargs

        assert kwargs['contrib_scores'].keys() == kwargs['hypothetical_contribs'].keys()

        print(f"Recieved args: {args}")
        print(f"Recieved kwargs: {kwargs}")
        return DummyModiscoResult()
