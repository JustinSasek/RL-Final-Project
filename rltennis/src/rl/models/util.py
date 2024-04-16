from dataclasses import asdict, dataclass, fields, is_dataclass

import torch
import torch.nn as nn


def nn_dataclass(cls):
    cls = dataclass(cls, kw_only=True, unsafe_hash=True)

    def init_wrapper(self, *args, init_func=cls.__init__, **kwargs):
        super(cls, self).__init__()
        init_func(self, *args, **kwargs)

    cls.__init__ = init_wrapper
    return cls


class ModelInterface(nn.Module):
    def config_dict(self) -> dict:
        assert is_dataclass(self)
        config_dict = asdict(self)  # type: ignore
        return config_dict

    def get_additional_losses(self) -> dict[str, torch.Tensor]:
        return {}

    @classmethod
    def load_config_dict(cls, config_dict: dict) -> "ModelInterface":
        model: ModelInterface = cls.__new__(cls)
        super(ModelInterface, model).__init__()  # type: ignore
        assert is_dataclass(model)
        for k, v in config_dict.items():
            if isinstance(v, dict):
                submodule_type = [f for f in fields(model) if f.name == k][0].type
                assert issubclass(submodule_type, ModelInterface)
                submodule: ModelInterface = submodule_type.load_config_dict(v)
                setattr(model, k, submodule)
            else:
                setattr(model, k, v)

        # call model super
        model.__post_init__()  # type: ignore
        return model  # type: ignore

    @classmethod
    def load_checkpoint(cls, path: str) -> tuple["ModelInterface", dict]:
        checkpoint = torch.load(path)
        model = cls.load_config_dict(checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state_dict"])
        del checkpoint["model_config"]
        del checkpoint["model_state_dict"]
        return model, checkpoint

    def save_checkpoint(self, path: str, **kwargs):
        model_config = self.config_dict()
        checkpoint = {
            "model_config": model_config,
            "model_state_dict": self.state_dict(),
            **kwargs,
        }
        torch.save(checkpoint, path)
