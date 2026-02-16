import torch

from torchmetrics import Metric


class PhysicalMetrics(Metric):
    """
    Physical metrics for generated motions.
    Expects joints with shape [B, F, J, 3] and a tensor/list of lengths [B].

    Metrics returned by compute():
      - skate_ratio: mean per-sequence skating ratio (unitless)
      - mean_penetration: mean per-sequence worst penetration depth (cm)
      - penetration: mean penetration over all frames (mm)
      - floating: mean floating over all frames (mm)
      - skating: mean foot sliding distance per frame pair (mm)
    """

    def __init__(self, dataset_name: str, dist_sync_on_step: bool = True) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.name = "Physical metrics"
        self.dataset_name = dataset_name  # used to pick foot joints for skating ratio

        # sequence-level counters
        self.add_state("count_seq", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state(
            "skate_ratio_sum", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state(
            "mean_penetration_sum", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )

        # frame-level accumulators (sum and counts of contributing items)
        self.add_state(
            "penetration_sum", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state(
            "penetration_count", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )

        self.add_state("floating_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state(
            "floating_count", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )

        self.add_state("skating_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("skating_count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    @staticmethod
    def _skating_ratio(
        joints: torch.Tensor, dataset_name: str, lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        joints: [B, F, J, 3]; returns [B] skating ratios in [0,1].
        Uses feet indices based on dataset. Ignores padded frames beyond lengths.
        """
        import numpy as np
        from scipy.ndimage import uniform_filter1d

        assert joints.ndim == 4, f"Expected [B,F,J,3], got {joints.shape}"
        J = joints.shape[2]
        if dataset_name == "humanml3d":
            foot_idx = [10, 11]
        elif dataset_name == "kit":
            foot_idx = [15, 20]
        else:
            # Try to infer by joint count
            if J == 22:
                foot_idx = [10, 11]
            elif J == 21:
                foot_idx = [15, 20]
            else:
                raise ValueError(f"Unsupported dataset_name {dataset_name} and J={J}")

        thresh_height = 0.05
        fps = 20.0
        thresh_vel = 0.50
        avg_window = 5

        verts_feet = joints[:, :, foot_idx, :].detach().cpu().numpy()  # [B, F, 2, 3]
        # planar velocities on XZ between consecutive frames
        verts_feet_plane_vel = (
            np.linalg.norm(
                verts_feet[:, 1:, :, [0, 2]] - verts_feet[:, :-1, :, [0, 2]], axis=-1
            )
            * fps
        )  # [B, F-1, 2]
        vel_avg = uniform_filter1d(
            verts_feet_plane_vel, axis=1, size=avg_window, mode="constant", origin=0
        )

        verts_feet_height = verts_feet[:, :, :, 1]  # [B, F, 2]
        feet_contact = np.logical_and(
            verts_feet_height[:, :-1, :] < thresh_height,
            verts_feet_height[:, 1:, :] < thresh_height,
        )  # [B, F-1, 2]
        skating = np.logical_and(feet_contact, verts_feet_plane_vel > thresh_vel)
        skating = np.logical_and(skating, vel_avg > thresh_vel)
        skating = np.logical_or(skating[:, :, 0], skating[:, :, 1])  # [B, F-1]

        # mask out padded tail beyond lengths (pairs up to L-1)
        L = lengths.detach().cpu().numpy().astype(int)
        B, Fm1 = skating.shape
        mask = np.zeros_like(skating, dtype=bool)
        for i in range(B):
            valid = max(0, int(L[i]) - 1)
            if valid > 0:
                mask[i, :valid] = True
        valid_counts = np.maximum(mask.sum(axis=1), 1)
        skating_ratio = (skating & mask).sum(axis=1) / valid_counts
        return torch.from_numpy(skating_ratio).to(joints.device)

    @staticmethod
    def _penetration_sum_count(
        joints: torch.Tensor, lengths: torch.Tensor
    ):
        """Mean penetration across all frames, measured in mm.
        Returns (sum, count_frames). joints: [B,F,J,3]."""
        B = joints.shape[0]
        lowest_heights = joints[..., 1].amin(dim=2)  # [B, F]
        tolerance = 0.005  # m
        total_sum = torch.tensor(0.0, device=joints.device)
        total_count = torch.tensor(0.0, device=joints.device)
        for i in range(B):
            L = int(lengths[i])
            if L <= 0:
                continue
            vals = (lowest_heights[i, :L] + tolerance).clamp_max(0) * -1000.0  # mm
            total_sum += vals.sum()
            total_count += torch.tensor(float(L), device=joints.device)
        return total_sum, total_count

    @staticmethod
    def _floating_sum_count(
        joints: torch.Tensor, lengths: torch.Tensor
    ):
        """Mean floating across all frames, measured in mm. Returns (sum, count_frames)."""
        B = joints.shape[0]
        lowest_heights = joints[..., 1].amin(dim=2)  # [B, F]
        tolerance = 0.005  # m
        total_sum = torch.tensor(0.0, device=joints.device)
        total_count = torch.tensor(0.0, device=joints.device)
        for i in range(B):
            L = int(lengths[i])
            if L <= 0:
                continue
            vals = (lowest_heights[i, :L] - tolerance).clamp_min(0) * 1000.0  # mm
            total_sum += vals.sum()
            total_count += torch.tensor(float(L), device=joints.device)
        return total_sum, total_count

    @staticmethod
    def _foot_sliding_sum_count(
        joints: torch.Tensor, lengths: torch.Tensor
    ):
        """Foot sliding per consecutive frame pair, in mm. Returns (sum, count_pairs)."""
        B = joints.shape[0]
        margin = 0.005  # m
        total_sum = torch.tensor(0.0, device=joints.device)
        total_count = torch.tensor(0.0, device=joints.device)
        for i in range(B):
            L = int(lengths[i])
            if L <= 1:
                continue
            ji = joints[i]  # [F, J, 3]
            for t in range(L - 1):
                # contact joint = lowest joint at frame t
                contact_idx = torch.argmin(ji[t, :, 1])
                if (
                    ji[t, contact_idx, 1] <= margin
                    and ji[t + 1, contact_idx, 1] <= margin
                ):
                    offset = (
                        ji[t + 1, contact_idx, [0, 2]] - ji[t, contact_idx, [0, 2]]
                    )  # [2]
                    skate_i = torch.norm(offset) * 1000.0  # mm
                else:
                    skate_i = torch.tensor(0.0, device=joints.device)
                total_sum += skate_i
                total_count += 1.0
        return total_sum, total_count

    @staticmethod
    def _mean_penetration_per_seq(
        joints: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """Worst penetration per sequence (min over joints and frames), in cm. Returns [B]."""
        B = joints.shape[0]
        res = []
        for i in range(B):
            L = int(lengths[i])
            if L <= 0:
                res.append(torch.tensor(0.0, device=joints.device))
                continue
            min_h = joints[i, :L, :, 1].amin()  # scalar (m)
            val_cm = torch.clamp(-min_h, min=0.0) * 100.0
            res.append(val_cm)
        return torch.stack(res, dim=0)

    def update(self, joints: torch.Tensor, lengths) -> None:
        """
        joints: [B, F, J, 3]
        lengths: list[int] or 1D tensor of shape [B]
        """
        if not torch.is_tensor(lengths):
            lengths = torch.tensor(lengths, device=joints.device)
        lengths = lengths.to(joints.device)

        B = joints.shape[0]
        self.count_seq += B

        # skating ratio (per sequence, unitless)
        skate_ratio = self._skating_ratio(joints, self.dataset_name, lengths)
        self.skate_ratio_sum += skate_ratio.sum().to(self.skate_ratio_sum.device)

        # mean penetration per sequence (cm)
        mean_pen = self._mean_penetration_per_seq(joints, lengths)
        self.mean_penetration_sum += mean_pen.sum().to(self.mean_penetration_sum.device)

        # penetration/floating/skating aggregated over frames
        pen_sum, pen_cnt = self._penetration_sum_count(joints, lengths)
        flo_sum, flo_cnt = self._floating_sum_count(joints, lengths)
        skat_sum, skat_cnt = self._foot_sliding_sum_count(joints, lengths)

        self.penetration_sum += pen_sum.to(self.penetration_sum.device)
        self.penetration_count += pen_cnt.to(self.penetration_count.device)
        self.floating_sum += flo_sum.to(self.floating_sum.device)
        self.floating_count += flo_cnt.to(self.floating_count.device)
        self.skating_sum += skat_sum.to(self.skating_sum.device)
        self.skating_count += skat_cnt.to(self.skating_count.device)

    def compute(self):
        # Avoid division by zero
        count_seq = torch.clamp(self.count_seq, min=1)
        pen_cnt = torch.clamp(self.penetration_count, min=1.0)
        flo_cnt = torch.clamp(self.floating_count, min=1.0)
        skat_cnt = torch.clamp(self.skating_count, min=1.0)

        metrics = {
            "skate_ratio": self.skate_ratio_sum / count_seq,
            "mean_penetration": self.mean_penetration_sum / count_seq,
            "penetration": self.penetration_sum / pen_cnt,
            "floating": self.floating_sum / flo_cnt,
            "skating": self.skating_sum / skat_cnt,
        }
        return metrics
