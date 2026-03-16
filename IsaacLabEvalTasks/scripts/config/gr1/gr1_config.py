# gr1_config.py
#
# Modality config for GR1 (upper-body: 2 arms + 2 hands) with a single ego-view camera.
# This matches your meta/modality.json slices:
#   state/action: 26 dims total = 7 (L arm) + 7 (R arm) + 6 (L hand) + 6 (R hand)
#   video: ego_view
#   annotation: human.action.task_description

from gr00t.configs.data.embodiment_configs import (
    register_modality_config,
    EmbodimentTag,
)
from gr00t.data.types import (
    ModalityConfig,
    ActionConfig,
    ActionRepresentation,
    ActionType,
    ActionFormat,
)

gr1_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "ego_view",
        ],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "left_arm",
            "right_arm",
            "left_hand",
            "right_hand",
        ],
        # These are joint angles in radians → sin/cos is usually the right choice.
        # If your hand values are not angles (e.g., normalized tendon commands), remove them here.
        sin_cos_embedding_keys=[
            "left_arm",
            "right_arm",
            "left_hand",
            "right_hand",
        ],
    ),
    "action": ModalityConfig(
        # Common choice: predict a 16-step action horizon like SO-100.
        # Change to your rollout horizon if needed.
        delta_indices=list(range(0, 16)),
        modality_keys=[
            "left_arm",
            "right_arm",
            "left_hand",
            "right_hand",
        ],
        action_configs=[
            # left_arm (7)
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key="left_arm",
            ),
            # right_arm (7)
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key="right_arm",
            ),
            # left_hand (6)
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key="left_hand",
            ),
            # right_hand (6)
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key="right_hand",
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "annotation.human.action.task_description",
        ],
    ),
}

register_modality_config(
    gr1_config,
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
)