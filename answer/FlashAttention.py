import torch, einops

FA_params = {
    "BQ": 32,
    "BK": 32
}

class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal=False):
        BQ = FA_params["BQ"]
        BK = FA_params["BK"]
        B = Q.shape[0]
        d = Q.shape[-1]
        NQ = Q.shape[-2]
        NK = K.shape[-2]
        TQ = NQ // BQ
        TK = NK // BK

        O_final = Q.new_zeros([B, NQ, d])
        L_final = Q.new_zeros([B, NQ])

        for i in range(TQ):
            Qi = Q[:, i * BQ: (i + 1) * BQ, :] # [batch_size, BQ, d]
            O = Qi.new_zeros([B, BQ, d]) # [batch_size, BQ, d]
            l = Qi.new_zeros([B, BQ])
            m = l.new_full([B, BQ], float('-inf'))

            for j in range(TK):
                Kj = K[:, j * BK: (j + 1) * BK, :] # [batch_size, BK, d]
                Vj = V[:, j * BK: (j + 1) * BK, :] # [batch_size, BK, d]
                Sij = Qi @ Kj.transpose(-2, -1) / (d ** 0.5) # [batch_size, BQ, BK]
                mnew = torch.max(m, torch.max(Sij, dim=-1).values) # [batch_size, BQ]
                Pij = torch.exp(Sij - mnew[:, :, None]) # [batch_size, BQ, BK]
                lnew = torch.exp(m - mnew) * l + torch.sum(Pij, dim=-1) # [batch_size, BQ]
                Onew = torch.exp(m - mnew)[:, :, None] * O + Pij @ Vj

                m = mnew
                l = lnew
                O = Onew

            O_final[:, i * BQ: (i + 1) * BQ, :] = O / l[:, :, None]
            L_final[:, i * BQ: (i + 1) * BQ] = m + torch.log(l)

        ctx.save_for_backward(Q, K, V, O_final, L_final)

        return O_final

    @staticmethod
    def backward(ctx, grad_O):
        raise NotImplementedError