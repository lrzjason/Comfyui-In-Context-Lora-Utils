from .InContextLoraUtils import AddMaskForICLora
from .InContextUtils import CreateContextWindow, ConcatContextWindow, AutoPatch

NODE_CLASS_MAPPINGS = {
    "AddMaskForICLora": AddMaskForICLora,
    "CreateContextWindow": CreateContextWindow,
    "ConcatContextWindow": ConcatContextWindow,
    "AutoPatch": AutoPatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AddMaskForICLora": "Add Mask For IC Lora",
    "CreateContextWindow": "Create Context Window",
    "ConcatContextWindow": "Concatenate Context Window",
    "AutoPatch": "Auto Patch",
}
