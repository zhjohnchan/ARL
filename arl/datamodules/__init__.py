from .cls_melinda_datamodule import CLSMELINDADataModule
from .irtr_roco_datamodule import IRTRROCODataModule
from .pretraining_medicat_datamodule import MedicatDataModule
from .pretraining_mimic_cxr_datamodule import MIMICCXRDataModule
from .pretraining_roco_datamodule import ROCODataModule
from .vqa_medvqa_2019_datamodule import VQAMEDVQA2019DataModule
from .vqa_medvqa_2021_datamodule import VQAMEDVQA2021DataModule
from .vqa_slack_datamodule import VQASLACKDataModule
from .vqa_vqa_rad_datamodule import VQAVQARADDataModule

_datamodules = {
    "medicat": MedicatDataModule,
    "roco": ROCODataModule,
    "mimic_cxr": MIMICCXRDataModule,
    "vqa_vqa_rad": VQAVQARADDataModule,
    "vqa_slack": VQASLACKDataModule,
    "vqa_medvqa_2019": VQAMEDVQA2019DataModule,
    "vqa_medvqa_2021": VQAMEDVQA2021DataModule,
    "cls_melinda": CLSMELINDADataModule,
    "irtr_roco": IRTRROCODataModule
}
