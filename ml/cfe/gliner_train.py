import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from gliner2 import AutoExtractor
from gliner2.training.trainer import ExtractorTrainer, TrainingConfig

model = AutoExtractor.from_pretrained("fastino/gliner2.5-multi-v1")
cfg = TrainingConfig(
    output_dir="runs/gliner-v2", num_epochs=2, max_train_samples=60000, batch_size=2, gradient_accumulation_steps=4, encoder_lr=1e-5, task_lr=5e-4,
    warmup_ratio=0.05, scheduler_type="cosine", fp16=True, eval_strategy="epoch", save_best=True,
    logging_steps=50, report_to_wandb=False,
)
ExtractorTrainer(model, cfg).train(train_data="data/gliner_train.jsonl", eval_data="data/gliner_dev.jsonl")
