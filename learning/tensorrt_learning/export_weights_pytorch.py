import struct

from loguru import logger
from torch import Tensor, nn


def export_weights(model: nn.Module):
    with open(f"{model._get_name()}.weights", "w") as f:
        f.write(f"{len(model.state_dict().keys())}\n")

        for name, weights in model.state_dict().items():
            weights: Tensor
            logger.info(f"exporting {name}: {weights.shape}")

            weights_1dims = weights.reshape(-1).cpu().numpy()
            f.write(f"{name} {len(weights_1dims)}")
            for i in weights_1dims:
                f.write(f" {struct.unpack('>f',float(i).hex())}")
            f.write("\n ")


def main():
    pass


if __name__ == "__main__":
    main()
