import hydra
import torch

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from tuner import QwenImageEditFinetuner


@hydra.main(version_base="v1.2", config_path="configs", config_name="default")
def finetune(cfgs: OmegaConf):

    # -- Initialize Training Framework -- #
    accelerator = Accelerator(
        mixed_precision=cfgs.mixed_precision,
        gradient_accumulation_steps=cfgs.gradient_accumulation,
        log_with=cfgs.log_with,
        project_config=ProjectConfiguration(
            project_dir=cfgs.project_dir,
            logging_dir=cfgs.logging_dir,
        ),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)],
    )
    if accelerator.is_main_process:
        accelerator.init_trackers(cfgs.project_name)

    logger = get_logger(__name__, log_level="INFO")
    logger.info(f"# -- Configs -- #")
    logger.info(OmegaConf.to_yaml(cfgs))

    # -- Initialize Dataset -- #
    trainset = instantiate(cfgs.trainset)
    valset = instantiate(cfgs.valset)

    logger.info(f"Num of train samples: {len(trainset)}")
    logger.info(f"Num of val samples: {len(valset)}")

    train_loader = DataLoader(
        trainset,
        batch_size=cfgs.train_batch_size,
        shuffle=False,
        num_workers=cfgs.train_num_workers,
    )
    val_loader = DataLoader(
        valset,
        batch_size=cfgs.val_batch_size,
        shuffle=False,
        num_workers=cfgs.val_num_workers,
    )

    train_loader: DataLoader = accelerator.prepare(train_loader)
    val_loader: DataLoader = accelerator.prepare(val_loader)

    # -- Initialize Finetuner -- #
    finetuner: QwenImageEditFinetuner = instantiate(cfgs.finetuner, device=accelerator.device, logger=logger)

    # -- Add Single LoRA -- #
    finetuner.register_styctrl(
        lora_target_modules=cfgs.styctrl.lora_target_modules,
        device=accelerator.device,
        dtype=finetuner.dtype,
    )
    finetuner.add_styctrl(
        rank=cfgs.styctrl.rank,
        adapter_name=cfgs.styctrl.adapter_name,
        proj_type=cfgs.styctrl.proj_type,
        bias=cfgs.styctrl.bias,
        checkpoint_path=cfgs.styctrl.checkpoint_path,
        device=accelerator.device,
        dtype=finetuner.dtype,
    )
    finetuner.activate_styctrl(cfgs.styctrl.adapter_name)

    # from safetensors.torch import save_file
    # states = {}
    # for n, p in finetuner.transformer.named_parameters():
    #     if cfgs.styctrl.adapter_name and "projector" in n:
    #         states[n] = p
    # save_file(states, "experiment/pretrained_weights/fluxkontext/s5.safetensors")

    # -- Start Training -- #
    finetuner.finetune(
        accelerator=accelerator,
        max_training_steps=cfgs.max_training_steps,
        train_loader=train_loader,
        val_loader=val_loader,
        adapter_name=cfgs.styctrl.adapter_name,
        generator=torch.Generator(device=accelerator.device).manual_seed(cfgs.seed),
        checkpointing_steps=cfgs.checkpointing_steps,
        validation_steps=cfgs.validation_steps,
        output_dir=cfgs.project_dir,
    )


if __name__ == "__main__":
    finetune()
