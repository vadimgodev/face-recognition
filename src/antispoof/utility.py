"""Utility functions for anti-spoofing model."""


def get_kernel(height: int, width: int) -> tuple:
    """Calculate kernel size based on input dimensions."""
    kernel_size = ((height + 15) // 16, (width + 15) // 16)
    return kernel_size


def parse_model_name(model_name: str) -> tuple:
    """
    Parse model filename to extract parameters.

    Example: "2.7_80x80_MiniFASNetV2.pth"
    Returns: (80, 80, "MiniFASNetV2", 2.7)

    Args:
        model_name: Model filename

    Returns:
        Tuple of (height, width, model_type, scale)
    """
    info = model_name.split('_')[0:-1]
    h_input, w_input = info[-1].split('x')
    model_type = model_name.split('.pth')[0].split('_')[-1]

    if info[0] == "org":
        scale = None
    else:
        scale = float(info[0])

    return int(h_input), int(w_input), model_type, scale
