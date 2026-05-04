# src/kinematics_torch.py
import torch

def energy(p, m):
    return torch.sqrt(p*p + m*m)

def sin_from_cos(mu):
    return torch.sqrt(torch.clamp(1.0 - mu*mu, min=0.0))

def cosphi(Ei, En, Em, pi, pn, pm, mu2, mu3):
    s2 = sin_from_cos(mu2)
    s3 = sin_from_cos(mu3)

    num = (
        En*Em - En*Ei - Em*Ei
        + pi*(pn*mu2 + pm*mu3)
        - pn*pm*mu2*mu3
    )
    den = pn*pm*s2*s3

    cosph = torch.where(den > 0.0, num/den, torch.full_like(num, 2.0))
    mask = (torch.abs(cosph) <= 1.0) & (den > 0.0)
    return cosph, mask

def mandelstam_s_t(Ei, En, Em, pi, pn, pm, mu2, mu3, m):
    s = 2.0 * ((En + Em)*Ei - pi*(pn*mu2 + pm*mu3))
    t = 2.0*m*m - 2.0*Ei*En + 2.0*pi*pn*mu2
    return s, t

def sqrt_one_minus_cosphi2(cosphi):
    return torch.sqrt(torch.clamp(1.0 - cosphi*cosphi, min=0.0))
