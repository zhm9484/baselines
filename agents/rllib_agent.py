from typing import Optional
from ray.rllib.agents.trainer import Trainer

from rllib_wrapper import build_rllib_config
from agents.neural_agent import NeuralAgent


class RLlibAgent(NeuralAgent):
    name = "RLlib_"

    agent: Optional[Trainer] = None
    state: Optional[dict] = None
    policy_id: str = 'policy_0'

    def _neural_init(self,
                     rllib_env=None,
                     trainer_wrapper=None,
                     checkpoint_path=None):
        if rllib_env and trainer_wrapper:
            rllib_config = build_rllib_config(self.config,
                                              rllib_env,
                                              init_ray=False)
            trainer_cls, _ = trainer_wrapper(self.config)
            self.agent: Trainer = trainer_cls(config=rllib_config,
                                              env=rllib_env)
            if checkpoint_path:
                self.agent.restore(checkpoint_path)
            self.state = {}

    def __call__(self, obs):
        assert self.agent
        outs = self.agent.compute_actions({0: obs},
                                          state=self.state,
                                          policy_id=self.policy_id)
        actions = outs[0]
        self.state = outs[1]
        return actions


if __name__ == "__main__":
    import nmmo
    from config import scale, bases
    import tasks

    class Test(scale.Debug, bases.Medium, nmmo.config.AllGameSystems):
        @property
        def SPAWN(self):
            return self.SPAWN_CONCURRENT

        TASKS = tasks.All
        NENT = 128
        NPOP = 1

    from rllib_wrapper import RLlibEnv, PPO
    config = Test()
    agent = RLlibAgent(
        config, 1, RLlibEnv, PPO,
        "experiments/Trainer/Medium_870d/checkpoint_001000/checkpoint-1000")

    from nmmo import Env
    env = Env(config)
    obs = env.reset()
    print(agent(obs[1]))
    print(agent(obs[1]))
    print(agent(obs[1]))
