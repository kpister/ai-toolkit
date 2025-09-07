"""

ostris/ai-toolkit on https://modal.com
Run training with the following command:
modal run run_modal.py --config-file-list-str=/root/ai-toolkit/config/whatever_you_want.yml

"""

import os
import modal


# define the volume for storing model outputs, using "creating volumes lazily": https://modal.com/docs/guide/volumes
# you will find your model, samples and optimizer stored in: https://modal.com/storage/your-username/main/flux-lora-models
model_volume = modal.Volume.from_name("flux-lora-models", create_if_missing=True)
data_volume = modal.Volume.from_name("qwen-data", create_if_missing=True)

# modal_output, due to "cannot mount volume on non-empty path" requirement
MOUNT_DIR = "/root/ai-toolkit/modal_output"  # modal_output, due to "cannot mount volume on non-empty path" requirement
GPU = "H200"

# define modal app
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    # install required system and pip packages, more about this modal approach: https://modal.com/docs/examples/dreambooth_app
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.8",
        "torchvision==0.23.0",
        index_url="https://download.pytorch.org/whl/cu128",
        gpu=GPU,
    )
    .pip_install(
        "python-dotenv",
        "diffusers[torch]",
        "transformers==4.56.0",
        "torch==2.8",
        "torchvision==0.23.0",
        "ftfy",
        "oyaml",
        "opencv-python",
        "albumentations",
        "safetensors",
        "lycoris-lora",
        "flatten_json",
        "pyyaml",
        "tensorboard",
        "kornia",
        "invisible-watermark",
        "einops",
        "accelerate",
        "toml",
        "pydantic",
        "omegaconf",
        # "k-diffusion",
        "open_clip_torch",
        "timm",
        "prodigyopt",
        "bitsandbytes",
        "hf_transfer",
        "lpips",
        "pytorch_fid",
        "optimum-quanto",
        "sentencepiece",
        "huggingface_hub",
        "peft",
        gpu=GPU,
    )
    .pip_install("torchao")
    .pip_install("controlnet_aux")
    .pip_install("flash-attn==2.8.3", extra_options="--no-build-isolation")
    .workdir("/root/ai-toolkit")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "DISABLE_TELEMETRY": "YES"})
    .add_local_dir("/Users/kaiser/home/work/ai-toolkit", remote_path="/root/ai-toolkit")
)

# create the Modal app with the necessary mounts and volumes
app = modal.App(
    name="qwen-lora-training-h200",
    image=image,
    # mounts=[code_mount],
    volumes={MOUNT_DIR: model_volume, "/qwen-data": data_volume},
)


def print_end_message(jobs_completed, jobs_failed):
    failure_string = (
        f"{jobs_failed} failure{'' if jobs_failed == 1 else 's'}"
        if jobs_failed > 0
        else ""
    )
    completed_string = (
        f"{jobs_completed} completed job{'' if jobs_completed == 1 else 's'}"
    )

    print("")
    print("========================================")
    print("Result:")
    if len(completed_string) > 0:
        print(f" - {completed_string}")
    if len(failure_string) > 0:
        print(f" - {failure_string}")
    print("========================================")


@app.function(
    gpu=GPU,
    timeout=7200 * 2,  # 4 hours, increase or decrease if needed
    cpu=16,
    memory=164 * 1024,  # 128GB RAM
)
def main(name: str | None = None):
    from toolkit.job import get_job

    # convert the config file list from a string to a list
    config_file = "config/examples/train_lora_qwen_image_edit_32gb.yaml"

    jobs_completed = 0
    jobs_failed = 0

    job = get_job(config_file, name)

    job.config["process"][0]["training_folder"] = MOUNT_DIR
    os.makedirs(MOUNT_DIR, exist_ok=True)
    print(f"Training outputs will be saved to: {MOUNT_DIR}")

    job.run()
    model_volume.commit()

    job.cleanup()
    jobs_completed += 1

    print_end_message(jobs_completed, jobs_failed)
