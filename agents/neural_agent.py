from nmmo.core.agent import Agent


class NeuralAgent(Agent):
    def __init__(self, config, idx, *args, **kwargs):
        super().__init__(config, idx)

        self._neural_init(*args, **kwargs)

    def _neural_init(self, *args, **kwargs):
        raise NotImplementedError
