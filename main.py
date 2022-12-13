import torch
from omegaconf import OmegaConf

import trainer


def main(config):
    torch.manual_seed(42)
    cur_trainer = getattr(trainer, config.general.trainer)(config)
    cur_trainer.setup_data()
    cur_trainer.setup_loaders()
    cur_trainer.setup_model()
    cur_trainer.setup_optim()
    cur_trainer.setup_loss()
    cur_trainer.train_loop()


if __name__ == "__main__":
    conf_cli = OmegaConf.from_cli()
    run_config = OmegaConf.merge(OmegaConf.load(conf_cli.config), conf_cli)
    main(run_config)
