"""
Litigation as query-by-committee.

A dispute is a point x in R^d. The court's rule is a direction u (unit vector)
and a sharpness k:  P(plaintiff wins | x) = sigmoid(k * u.x).
Large k = determinate court; small k = noisy court.

Parties share one public model of the court: a Bayesian logistic regression
(Laplace approximation) trained on adjudicated cases only. Each side's view of
a new dispute is an independent draw from that model's posterior over the
margin. They go to trial iff the plaintiff's estimate of its win probability
exceeds the defendant's estimate by more than c (total trial cost / stakes).
That is the Landes-Posner-Gould / Priest-Klein condition, with the dispersion
of party beliefs made endogenous to how much the system has learned.

With c = 0 and sign disagreement this is the two-member query-by-committee
filter of Seung, Opper & Sompolinsky (1992).
"""
import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


class PublicModel:
    """Bayesian logistic regression, Laplace approximation, optional forgetting."""

    def __init__(self, d, prior_prec=0.05):
        self.d = d
        self.alpha = prior_prec
        self.X = np.empty((0, d))
        self.y = np.empty(0)
        self.w = np.empty(0)          # observation weights (1, or decayed)
        self.mu = np.zeros(d)
        self.Sigma = np.eye(d) / prior_prec

    def add(self, X, y):
        if len(y) == 0:
            return
        self.X = np.vstack([self.X, X])
        self.y = np.concatenate([self.y, y])
        self.w = np.concatenate([self.w, np.ones(len(y))])

    def decay(self, lam):
        if lam >= 1.0 or len(self.w) == 0:
            return
        self.w *= lam
        keep = self.w > 1e-3
        if not keep.all():
            self.X, self.y, self.w = self.X[keep], self.y[keep], self.w[keep]

    def _obj(self, mu):
        z = self.X @ mu
        ll = np.sum(self.w * (self.y * z - np.logaddexp(0, z)))
        return ll - 0.5 * self.alpha * mu @ mu

    def fit(self, iters=40, tol=1e-7):
        d = self.d
        if len(self.y) == 0:
            self.mu = np.zeros(d)
            self.Sigma = np.eye(d) / self.alpha
            return
        mu = self.mu.copy()
        f = self._obj(mu)
        for _ in range(iters):
            p = sigmoid(self.X @ mu)
            g = self.X.T @ (self.w * (self.y - p)) - self.alpha * mu
            H = (self.X * (self.w * p * (1 - p))[:, None]).T @ self.X + self.alpha * np.eye(d)
            step = np.linalg.solve(H, g)
            t = 1.0
            while t > 1e-6:
                f_new = self._obj(mu + t * step)
                if f_new >= f:
                    break
                t *= 0.5
            mu = mu + t * step
            if abs(f_new - f) < tol * (1 + abs(f)):
                f = f_new
                break
            f = f_new
        p = sigmoid(self.X @ mu)
        H = (self.X * (self.w * p * (1 - p))[:, None]).T @ self.X + self.alpha * np.eye(d)
        self.mu = mu
        self.Sigma = np.linalg.inv(H)

    def margin_stats(self, X):
        m = X @ self.mu
        v = np.einsum("ij,jk,ik->i", X, self.Sigma, X)
        return m, np.maximum(v, 0)

    def predict(self, X):
        """Posterior-predictive win probability (probit-moderated)."""
        m, v = self.margin_stats(X)
        return sigmoid(m / np.sqrt(1 + np.pi * v / 8))


def party_views(model, X, rng, shared=0.0):
    """Each side's estimate of plaintiff's win probability.

    shared in [0,1): fraction of belief dispersion the two sides have in common
    (0 = independent posterior draws; ->1 = both sides hold the same view).
    """
    m, v = model.margin_stats(X)
    s = np.sqrt(v)
    common = rng.standard_normal(len(m))
    eP = np.sqrt(shared) * common + np.sqrt(1 - shared) * rng.standard_normal(len(m))
    eD = np.sqrt(shared) * common + np.sqrt(1 - shared) * rng.standard_normal(len(m))
    return sigmoid(m + s * eP), sigmoid(m + s * eD)


def rotate_toward_random(u, eta, rng):
    """One step of a random walk on the sphere, step size eta (radians, approx)."""
    xi = rng.standard_normal(len(u))
    xi -= (xi @ u) * u
    xi /= np.linalg.norm(xi)
    v = np.cos(eta) * u + np.sin(eta) * xi
    return v / np.linalg.norm(v)


def run(
    d=20, k=5.0, c=0.3, N=2000, T=150, seed=0,
    selection="pk",        # "pk" | "random" (needs n_schedule) | "all"
    n_schedule=None,       # trials per period for selection="random"
    drift=0.0,             # radians per period of random-walk rotation of u
    forget=1.0,            # per-period weight decay on precedent (1 = none)
    shared=0.0,
    n_test=20000, n_seed_cases=40, prior_prec=0.05,
    keep_cases=False, u0=None, trickle=0.0,
    jump=None,             # dict(W=40, z=3.5, keep=0.2): abrupt revision when recent outcomes surprise
    shock=None,            # dict(t=800, angle_deg=30, announced=True, keep=0.1): one-off change in the law
):
    rng = np.random.default_rng(seed)
    if u0 is None:
        u = rng.standard_normal(d); u /= np.linalg.norm(u)
    else:
        u = np.array(u0, float); u /= np.linalg.norm(u)
    model = PublicModel(d, prior_prec)
    Xt = rng.standard_normal((n_test, d))

    # a few founding precedents, randomly selected
    X0 = rng.standard_normal((n_seed_cases, d))
    y0 = (rng.random(n_seed_cases) < sigmoid(k * X0 @ u)).astype(float)
    model.add(X0, y0); model.fit()

    out = {key: [] for key in [
        "trials", "trial_rate", "cum_trials", "oracle_err", "sign_err", "pop_acc",
        "pop_ceiling", "tried_acc", "tried_ceiling", "tried_pwin", "angle_deg",
        "khat", "mean_disagree", "self_unc"]}
    cases = {"X": [], "y": [], "t": [], "pstar": []} if keep_cases else None
    cum = n_seed_cases
    recent = []            # (excess log-loss, its variance under the model) for recent tried cases
    resets = []

    for t in range(T):
        if drift > 0:
            u = rotate_toward_random(u, drift, rng)
        if shock is not None and t == shock["t"]:
            u = rotate_toward_random(u, np.radians(shock["angle_deg"]), rng)
            if shock["announced"]:           # everyone knows old precedent is now less reliable
                model.w *= shock["keep"]; model.fit()
        X = rng.standard_normal((N, d))
        pstar = sigmoid(k * X @ u)

        if selection == "pk":
            pP, pD = party_views(model, X, rng, shared)
            tried = (pP - pD) > c
            if trickle > 0:   # cases tried regardless of party beliefs
                tried |= rng.random(N) < trickle
            out["mean_disagree"].append(float(np.mean(np.abs(pP - pD))))
        elif selection == "random":
            n = int(n_schedule[t])
            tried = np.zeros(N, bool)
            tried[rng.choice(N, size=min(n, N), replace=False)] = True
            out["mean_disagree"].append(np.nan)
        else:
            tried = np.ones(N, bool)
            out["mean_disagree"].append(np.nan)

        Xn = X[tried]
        yn = (rng.random(len(Xn)) < pstar[tried]).astype(float)

        # score the pre-trial public model on the cases that got tried
        if len(yn) > 0:
            ph = model.predict(Xn)
            out["tried_acc"].append(float(np.mean((ph > 0.5) == (yn > 0.5))))
            out["tried_ceiling"].append(float(np.mean(np.maximum(pstar[tried], 1 - pstar[tried]))))
            out["tried_pwin"].append(float(np.mean(yn)))
        else:
            out["tried_acc"].append(np.nan); out["tried_ceiling"].append(np.nan)
            out["tried_pwin"].append(np.nan)

        if jump is not None and len(yn) > 0:
            q = np.clip(model.predict(Xn), 1e-4, 1 - 1e-4)
            loss = -(yn * np.log(q) + (1 - yn) * np.log(1 - q))
            ent = -(q * np.log(q) + (1 - q) * np.log(1 - q))
            var = q * (1 - q) * np.log(q / (1 - q)) ** 2
            recent.extend(zip(loss - ent, var))
            recent = recent[-jump["W"]:]
            if len(recent) >= jump["W"]:
                sv = sum(v for _, v in recent)
                zstat = sum(e for e, _ in recent) / np.sqrt(sv + 1e-9)
                if zstat > jump["z"]:
                    model.w *= jump["keep"]      # old precedent is treated as stale
                    recent = []
                    resets.append(t)

        if keep_cases and len(yn) > 0:
            cases["X"].append(Xn); cases["y"].append(yn)
            cases["t"].append(np.full(len(yn), t)); cases["pstar"].append(pstar[tried])

        model.decay(forget)
        model.add(Xn, yn)
        if len(yn) > 0 or forget < 1.0:
            model.fit()
        cum += len(yn)

        # population metrics against the current court
        pt = sigmoid(k * Xt @ u)
        ph = model.predict(Xt)
        yt = rng.random(n_test) < pt
        out["trials"].append(int(len(yn)))
        out["trial_rate"].append(len(yn) / N)
        out["cum_trials"].append(cum)
        out["oracle_err"].append(float(np.mean(np.abs(ph - pt))))
        out["sign_err"].append(float(np.mean((ph > 0.5) != (pt > 0.5))))
        out["pop_acc"].append(float(np.mean((ph > 0.5) == yt)))
        out["pop_ceiling"].append(float(np.mean(np.maximum(pt, 1 - pt))))
        nm = np.linalg.norm(model.mu)
        cosang = np.clip(model.mu @ u / max(nm, 1e-12), -1, 1)
        out["angle_deg"].append(float(np.degrees(np.arccos(cosang))))
        out["khat"].append(float(nm))
        _, vt = model.margin_stats(Xt)
        out["self_unc"].append(float(np.mean(np.sqrt(vt) * ph * (1 - ph))))

    out = {key: np.array(val) for key, val in out.items()}
    out["u_final"] = u
    out["resets"] = np.array(resets)
    out["model"] = model
    if keep_cases:
        out["cases"] = {key: (np.concatenate(val) if val else np.empty(0)) for key, val in cases.items()}
    return out
