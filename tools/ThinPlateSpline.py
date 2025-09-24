import torch


class ThinPlateSpline:
    def __init__(self, source_points, target_points, reg_lambda=1e-6, device='cuda'):
        self.device = device
        self.source_points = source_points.to(device)
        self.target_points = target_points.to(device)
        self.N, self.M, self.C = self.source_points.shape
        self.reg_lambda = reg_lambda
        self.weights = self.compute_weights()

    def compute_weights(self):
        K = self.pairwise_tps_kernel(self.source_points, self.source_points)
        ones = torch.ones((self.N, self.M, 1), device=self.device)
        P = torch.cat([self.source_points, ones], dim=2)

        L_top = torch.cat([K, P], dim=2)
        P_T = P.transpose(1, 2)
        zeros = torch.zeros((self.N, self.C + 1, self.C + 1), device=self.device)
        L_bottom = torch.cat([P_T, zeros], dim=2)
        L = torch.cat([L_top, L_bottom], dim=1)

        I = torch.eye(L.size(-1), device=self.device).expand(self.N, -1, -1)
        L_reg = L + self.reg_lambda * I

        zeros_Y = torch.zeros((self.N, self.C + 1, self.C), device=self.device)
        Y = torch.cat([self.target_points, zeros_Y], dim=1)

        # weights = torch.linalg.solve(L_reg, Y)
        weights = torch.matmul(torch.linalg.pinv(L_reg), Y)
        return weights

    def pairwise_tps_kernel(self, X, Y):
        dist_sq = torch.cdist(X, Y, p=2) ** 2
        return dist_sq * torch.log(dist_sq + 1e-8)

    def transform(self, points):
        points = points.to(self.device)
        K = self.pairwise_tps_kernel(points, self.source_points)
        ones = torch.ones((points.size(0), points.size(1), 1), device=self.device)
        P = torch.cat([points, ones], dim=2)
        L = torch.cat([K, P], dim=2)
        return torch.matmul(L, self.weights)
