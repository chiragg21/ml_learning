import os
import platform
import torch

from .tracking_schemas import EnvironmentInfo


def get_environment_info() -> EnvironmentInfo:
    cuda_available = torch.cuda.is_available()

    return EnvironmentInfo(
        python_version=platform.python_version(),
        pytorch_version=torch.__version__,
        cuda_version=torch.version.cuda if cuda_available else None,
        device="cuda" if cuda_available else "cpu",
        device_name=(
            torch.cuda.get_device_name(0)
            if cuda_available
            else None
        ),
        operating_system=platform.platform(),
        cpu=platform.processor(),
        cpu_count=os.cpu_count(),
    )

if __name__=='__main__':
    print(get_environment_info())