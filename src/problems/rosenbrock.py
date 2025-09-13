import numpy as np


class RosenbrockProblem:
  """This class encapsulates the Rosenbrock Function optimization problem"""

  def __init__(self, dimensions=2, a=1, b=100):
    """
    Initialize the Rosenbrock problem
    :param dimensions: number of dimensions (default: 2)
    :param a: parameter a (default: 1)
    :param b: parameter b (default: 100)
    """
    # initialize instance variables:
    self.dimensions = dimensions
    self.a = a
    self.b = b
    self.bounds = [(-5, 5)] * dimensions  # Default bounds for each dimension

  def __len__(self):
    """
    :return: the number of dimensions (variables) in the problem
    """
    return self.dimensions

  def fitness(self, solution):
    """
    Calculates the Rosenbrock function value for the given solution
    :param solution: a list of real values representing the point in n-dimensional space
    :return: the calculated Rosenbrock function value (to be minimized)
    """
    if len(solution) != self.dimensions:
      raise ValueError(f"Solution must have {self.dimensions} dimensions")

    x = np.array(solution)
    result = 0
    for i in range(len(x) - 1):
      result += self.b * (x[i + 1] - x[i] ** 2) ** 2 + (self.a - x[i]) ** 2
    return result,

  def plotData(self, solution):
    """
    Prints information about the solution and its fitness
    :param solution: a list of real values representing the point
    """
    fitness_value = self.fitness(solution)

    print("Rosenbrock Function Solution:")
    print(f" - Dimensions: {self.dimensions}")
    print(f" - Parameters: a = {self.a}, b = {self.b}")
    print(f" - Solution point: {solution}")
    print(f" - Function value: {fitness_value}")

    # For 2D case, provide additional information
    if self.dimensions == 2:
      x, y = solution
      print(f" - Components:")
      print(f"   * (a - x)² = {((self.a - x) ** 2):.6f}")
      print(f"   * b*(y - x²)² = {(self.b * (y - x ** 2) ** 2):.6f}")

    print(f" - Global minimum at: {[self.a] * self.dimensions}")
    print(f" - Global minimum value: 0.0")
