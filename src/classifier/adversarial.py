"""FGM (Fast Gradient Method) adversarial training trên embedding layer.

    r_adv = epsilon * grad(x_emb) / ||grad(x_emb)||
    L_total = L(x_emb, y) + L(x_emb + r_adv, y)

Cách dùng: Seq2SeqTrainer chuẩn chỉ backward 1 lần/step. FGMSeq2SeqTrainer
override training_step để: (1) backward loss thường, (2) dịch chuyển các
tham số embedding theo r_adv, (3) backward loss adversarial trên embedding
đã nhiễu, (4) khôi phục embedding gốc trước khi optimizer.step(). Gradient
của 2 lần backward cộng dồn tự nhiên (không zero_grad giữa chừng).

Áp dụng trên tham số có emb_name trong tên (mặc định "shared" — tên embedding
dùng chung input/output của kiến trúc T5) để tương thích cả khi model bị
nn.DataParallel bọc ngoài (Kaggle 2x T4): named_parameters() lúc đó trả về
tên dạng "module.shared.weight", vẫn khớp substring "shared".
"""

from __future__ import annotations

import torch
from transformers import Seq2SeqTrainer


class FGMSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, *args, use_fgm: bool = True, fgm_epsilon: float = 1.0, fgm_emb_name: str = "shared", **kwargs):
        super().__init__(*args, **kwargs)
        self.use_fgm = use_fgm
        self.fgm_epsilon = fgm_epsilon
        self.fgm_emb_name = fgm_emb_name

    def _fgm_attack(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        backup: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if not param.requires_grad or self.fgm_emb_name not in name:
                continue
            if param.grad is None:
                continue
            norm = torch.norm(param.grad)
            if norm == 0 or not torch.isfinite(norm):
                continue
            backup[name] = param.data.clone()
            param.data.add_(self.fgm_epsilon * param.grad / norm)
        return backup

    @staticmethod
    def _fgm_restore(model: torch.nn.Module, backup: dict[str, torch.Tensor]) -> None:
        if not backup:
            return
        for name, param in model.named_parameters():
            if name in backup:
                param.data.copy_(backup[name])

    def _compute_loss(self, model, inputs, num_items_in_batch=None):
        with self.compute_loss_context_manager():
            try:
                return self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
            except TypeError:
                return self.compute_loss(model, inputs)

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)

        loss = self._compute_loss(model, inputs, num_items_in_batch)
        if self.args.n_gpu > 1:
            loss = loss.mean()
        self.accelerator.backward(loss)

        if self.use_fgm:
            backup = self._fgm_attack(model)
            if backup:
                loss_adv = self._compute_loss(model, inputs, num_items_in_batch)
                if self.args.n_gpu > 1:
                    loss_adv = loss_adv.mean()
                self.accelerator.backward(loss_adv)
                self._fgm_restore(model, backup)

        return loss.detach()
