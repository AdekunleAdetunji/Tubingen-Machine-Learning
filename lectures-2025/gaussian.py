import dataclasses
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

    def __add__(self, other: Float[Array, "D "] | float) -> Self:
        """
        Affine maps of Gaussian RVs are Gaussian RVs shift of
        Gaussian RVs by a constant
        """
        return Gaussian(mu=(self.mu + other), Sigma=self.Sigma)
