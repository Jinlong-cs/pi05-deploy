from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentPreset:
    name: str
    repo_id: str
    env_type: str | None
    eval_episodes: int
    train_max_episodes: int
    batch_size: int
    num_workers: int
    description: str
    model_family: str = "pi05"
    model_repo_id: str | None = None
    task_key: str = "task"


PRESETS = {
    "pi05_aloha_public": ExperimentPreset(
        name="pi05_aloha_public",
        repo_id="lerobot/aloha_sim_insertion_human_image",
        env_type="aloha",
        eval_episodes=10,
        train_max_episodes=40,
        batch_size=1,
        num_workers=0,
        description="Public ALOHA preset for PI0.5 Orin deployment.",
        model_family="pi05",
        model_repo_id="lerobot/pi05_base",
    ),
    "pi05_libero_public": ExperimentPreset(
        name="pi05_libero_public",
        repo_id="physical-intelligence/libero",
        env_type="libero",
        eval_episodes=20,
        train_max_episodes=200,
        batch_size=1,
        num_workers=0,
        description="Public LIBERO preset for PI0.5 Orin deployment.",
        model_family="pi05",
        model_repo_id="lerobot/pi05_libero",
    ),
}

DEFAULT_PRESET = "pi05_aloha_public"


def get_preset(name: str) -> ExperimentPreset:
    try:
        return PRESETS[name]
    except KeyError as exc:
        available = ", ".join(sorted(PRESETS))
        raise ValueError(f"Unknown preset '{name}'. Available presets: {available}") from exc
