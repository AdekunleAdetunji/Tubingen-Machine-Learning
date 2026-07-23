import dataclasses
import functools
import jax
import jax.numpy as jnp

from jaxtyping import Float, Array
from typing import Self


@dataclasses.dataclass
class Gaussian:
    # Gaussian Distribution with mean and covariance function
    mu: Float[Array, "D"]
    Sigma: Float[Array, "D D"]

    def log_pdf(self, x: Float[Array, "D "]) -> float:
        return -0.5 * (
            (len(self.mu) * jnp.log(2 * jnp.pi))
            + (jnp.linalg.slogdet(self.Sigma)[1])
            + ((x - self.mu) @ jnp.linalg.solve(self.Sigma, x - self.mu))
        )

    def pdf(self, x: Float[Array, "D "]) -> float:
        return jnp.exp(self.log_pdf(x))

    def precision(self):
        """
        Precision matrix
        Be careful not to use matrix inverses
        """
        return jnp.linalg.inv(self.Sigma)

    def mean_precision(self):
        """Precision adjusted mean"""
        return self.precision() @ self.mu

    def __mul__(self, other: Self) -> Self:
        """
        Product of Gaussian pdfs
        Multiplication of two gaussian PDFs (not RVs)
        """
        Sigma = jnp.linalg.inv(self.precision() + other.precision())
        mu = Sigma @ (self.mean_precision() + other.mean_precision())
        return Gaussian(mu=mu, Sigma=Sigma)

    @functools.singledispatchmethod
    def __add__(self, other: Float[Array, "D "] | float) -> Self:
        """
        Affine maps of Gaussian RVs are Gaussian RVs shift of
        Gaussian RVs by a constant
        """
        return Gaussian(mu=(self.mu + other), Sigma=self.Sigma)

    def __rmatmul__(self, A: Float[Array, "N D"]) -> Self:
        """
        Linear maps of Gaussian RVs are Gaussian RVs

        Returns: p(A @ x) = N(A @ x; A @ mu, A @ Sigma @ A.T)
        """
        return Gaussian(mu=A @ self.mu, Sigma=A @ self.Sigma @ A.T)

    def __getitem__(self, key):
        """Compute marginals"""
        return Gaussian(mu=self.mu[key], Sigma=self.Sigma[key, key])

    def condition(
        self, A: Float[Array, "N D"], y: Float[Array, "N"], Lambda: Float[Array, "N N"]
    ) -> Self:
        """
        Linear conditionals of gaussian RVs are Gaussian RVs

        Returns: p(self | y) = N(y; A @ self, lambda) * self / p(y)
        """
        Gram = A @ self.Sigma @ A.T + Lambda
        L = jax.scipy.linalg.cho_factor(Gram, lower=True)
        mu = self.mu + self.Sigma @ A.T @ jax.scipy.linalg.cho_solve(L, y - A @ self.mu)
        Sigma = self.Sigma - self.Sigma @ A.T @ jax.scipy.linalg.cho_solve(
            L, A @ self.Sigma
        )
        return Gaussian(mu=mu, Sigma=Sigma)


@Gaussian.__add__.register
def _(self, other: Gaussian) -> Gaussian:
    """Sum of two independent Gaussian RVs"""
    return Gaussian(mu=self.mu + other.mu, Sigma=self.Sigma + other.Sigma)
