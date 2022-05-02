from __future__ import annotations

import math
import random
import time
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass
class VRPInstanceConfig:
    n_customers: int
    n_vehicles: int
    vehicle_capacity: float
    demands: list[float]
    xs: list[float]
    ys: list[float]

    @classmethod
    def from_file(cls, filename: str) -> VRPInstanceConfig:
        n_customers, n_vehicles, vehicle_capacity = 0, 0, 0
        demands, xs, ys = [], [], []
        with open(filename, "r") as f:
            for i, line in enumerate(f.readlines()):
                tokens = line.strip().split(" ")
                if len(tokens) != 3:
                    continue
                if i == 0:
                    n_customers, n_vehicles, vehicle_capacity = (
                        int(tokens[0]),
                        int(tokens[1]),
                        float(tokens[2]),
                    )
                else:
                    demands.append(float(tokens[0]))
                    xs.append(float(tokens[1]))
                    ys.append(float(tokens[2]))

        return VRPInstanceConfig(
            n_customers, n_vehicles, vehicle_capacity, demands, xs, ys
        )


@dataclass
class VRPSolution:
    routes: list[list[int]]
    config: VRPInstanceConfig

    TEMPORARY_VEHICLE_PENALTY = 1e5

    def __repr__(self):
        lines = []
        for i, route in enumerate(self.routes):
            if i != len(self.routes) - 1:
                lines.append(
                    f"Vehicle {i} ({self.get_route_capacity(route)}): "
                    + " -> ".join(map(str, route))
                )
            else:
                lines.append(f'Unassigned: {",".join(map(str, route))}')
        return "\n".join(lines)

    def as_file_str(self, assert_temporary_route=True) -> str:
        lines = [f"{self.get_objective_value():.1f} 0"]
        for i, route in enumerate(self.routes):
            if i == len(self.routes) - 1:
                if len(route) != 0 and assert_temporary_route:
                    raise ValueError("Temporary route is not empty")
                continue
            lines.append(" ".join(["0", *map(str, route), "0"]))
        return "\n".join(lines)

    def as_solution_str(self) -> str:
        return " ".join(["0", *self.as_file_str().split("\n")[1:]])

    def export(self, filename: str):
        with open(filename, "w") as f:
            f.write(self.as_file_str(False))

    def get_objective_value(self, penalize_temp_route=True) -> float:
        objective_value = 0.0
        for i, route in enumerate(self.routes):
            if (
                i != len(self.routes) - 1
                and self.get_route_capacity(route) > self.config.vehicle_capacity
            ):
                return float("inf")
            distance = self.get_route_distance(route)
            if i == len(self.routes) - 1 and penalize_temp_route:
                distance *= self.TEMPORARY_VEHICLE_PENALTY
            objective_value += distance
        return objective_value

    def get_route_capacity(self, route: list[int]) -> float:
        capacity = 0
        for i in route:
            capacity += self.config.demands[i]
        return capacity

    def get_distance_between(self, i: int, j: int):
        return math.sqrt(
            (self.config.xs[i] - self.config.xs[j]) ** 2
            + (self.config.ys[i] - self.config.ys[j]) ** 2
        )

    def get_route_distance(self, route: list[int]) -> float:
        if len(route) == 0:
            return 0
        distance = self.get_distance_between(0, route[0])
        for i in range(len(route) - 1):
            distance += self.get_distance_between(route[i], route[i + 1])
        distance += self.get_distance_between(route[-1], 0)
        return distance

    def copy(self) -> VRPSolution:
        # https://stackoverflow.com/a/35008777
        return VRPSolution(list(map(list, self.routes)), self.config)

    def neighbor_two_exchange(self) -> VRPSolution:
        solution = self.copy()

        # pick a random route
        route_i_index = random.randint(0, len(solution.routes) - 1)
        route_i = solution.routes[route_i_index]
        if len(route_i) < 2:
            return solution
        # swap two random customers within the route and reverse the ordering
        i = random.randint(0, len(route_i) - 1)
        j = random.randint(i + 1, len(route_i))
        route_i[i:j] = reversed(route_i[i:j])
        solution.routes[route_i_index] = route_i
        return solution

    def neighbor_or_exchange(self) -> VRPSolution:
        solution = self.copy()

        # pick random routes
        route_i_index = random.randint(0, len(solution.routes) - 1)
        route_j_index = random.randint(0, len(solution.routes) - 1)
        route_i = solution.routes[route_i_index]
        route_j = solution.routes[route_j_index]

        if len(route_i) == 0 or len(route_j) == 0:
            return solution

        n_customers = random.randint(1, min(3, len(route_i)))
        i = random.randint(0, len(route_i) - n_customers)

        spliced = route_i[i : i + n_customers]
        route_i = route_i[:i] + route_i[i + n_customers :]

        if route_i_index == route_j_index:
            j = random.randint(0, len(route_i))
            route_i = route_i[:j] + spliced + route_i[j:]
            solution.routes[route_i_index] = route_i
        else:
            j = random.randint(0, len(route_j))
            route_j = route_j[:j] + spliced + route_j[j:]
            solution.routes[route_i_index] = route_i
            solution.routes[route_j_index] = route_j
        return solution

    def neighbor_relocation(self) -> VRPSolution:
        solution = self.copy()
        # pick random routes
        route_i_index = random.randint(0, len(solution.routes) - 1)
        route_j_index = random.randint(0, len(solution.routes) - 1)
        route_i = solution.routes[route_i_index]
        route_j = solution.routes[route_j_index]

        if len(route_i) == 0:
            return solution

        i = random.randint(0, len(route_i) - 1)
        customer = route_i.pop(i)
        j = random.randint(0, len(route_j))
        route_i.insert(j, customer)
        solution.routes[route_i_index] = route_i
        solution.routes[route_j_index] = route_j

        return solution

    def neighbor_exchange(self) -> VRPSolution:
        solution = self.copy()
        # pick random routes
        route_i_index = random.randint(0, len(solution.routes) - 1)
        route_j_index = random.randint(0, len(solution.routes) - 1)
        route_i = solution.routes[route_i_index]
        route_j = solution.routes[route_j_index]
        if len(route_i) == 0 or len(route_j) == 0 or route_i_index == route_j_index:
            return solution

        i = random.randint(0, len(route_i) - 1)
        j = random.randint(0, len(route_j) - 1)

        customer_i = route_i[i]
        customer_j = route_j[j]
        route_i[i] = customer_j
        route_j[j] = customer_i
        solution.routes[route_i_index] = route_i
        solution.routes[route_j_index] = route_j
        return solution

    def neighbor_crossover(self) -> VRPSolution:
        solution = self.copy()
        # pick random routes
        route_i_index = random.randint(0, len(solution.routes) - 1)
        route_j_index = random.randint(0, len(solution.routes) - 1)
        route_i = solution.routes[route_i_index]
        route_j = solution.routes[route_j_index]
        if len(route_i) == 0 or len(route_j) == 0 or route_i_index == route_j_index:
            return solution

        i = random.randint(0, len(route_i) - 1)
        j = random.randint(0, len(route_j) - 1)

        route_i_tail = route_i[i:]
        route_j_tail = route_j[j:]
        route_i = route_i[:i] + route_j_tail
        route_j = route_j[:j] + route_i_tail
        solution.routes[route_i_index] = route_i
        solution.routes[route_j_index] = route_j
        return solution

    def neighbor_portfolio(self) -> VRPSolution:
        solutions = [
            self.neighbor_two_exchange(),
            self.neighbor_relocation(),
            self.neighbor_or_exchange(),
            self.neighbor_crossover(),
            self.neighbor_exchange(),
        ]
        return min(solutions, key=lambda s: s.get_objective_value())

    def neighbor_random(self) -> VRPSolution:
        solutions = [
            # self.neighbor_two_exchange,
            self.neighbor_or_exchange,
            # self.neighbor_relocation,
            # self.neighbor_crossover,
            # self.neighbor_exchange,
        ]
        get_solution = random.choice(solutions)
        return min(
            [get_solution() for _ in range(len(solutions))],
            key=lambda s: s.get_objective_value(),
        )


class VRPInstance:
    def __init__(self, filename: str):
        self.config = VRPInstanceConfig.from_file(filename)

    def generate_initial_routes(self) -> VRPSolution:
        customers = sorted(
            [c for c in range(1, self.config.n_customers)],
            key=lambda c: self.config.demands[c],
            reverse=True,
        )

        routes = [[] for _ in range(self.config.n_vehicles)]
        routes_capacity = [0 for _ in range(self.config.n_vehicles)]
        unassigned = []

        for c in customers:
            is_assigned = False
            for r in range(self.config.n_vehicles):
                if (
                    routes_capacity[r] + self.config.demands[c]
                    <= self.config.vehicle_capacity
                ):
                    routes_capacity[r] += self.config.demands[c]
                    routes[r].append(c)
                    is_assigned = True
                    break

            if not is_assigned:
                unassigned.append(c)
        routes.append(unassigned)
        return VRPSolution(routes, self.config)

    # https://www.hindawi.com/journals/mpe/2019/2358258/fig3/
    def local_search(
        self,
        initial_solution: VRPSolution,
        neighbor: Callable[[VRPSolution], VRPSolution],
        temperature: float,
        max_iter: int,
    ) -> VRPSolution:
        best_solution = initial_solution
        best_objective = initial_solution.get_objective_value()

        current_solution = best_solution
        current_objective = best_objective

        def should_accept(next_objective) -> bool:
            delta = next_objective - current_objective
            if delta <= 0:
                return True
            if random.random() < math.exp(-delta / temperature):
                return True
            return False

        n_iter = 0
        while n_iter < max_iter:
            next_solution = neighbor(current_solution)
            next_objective = next_solution.get_objective_value()

            if should_accept(next_objective):
                current_solution = next_solution
                current_objective = next_objective

                if current_objective < best_objective:
                    best_solution = current_solution
                    best_objective = current_objective
                    n_iter = 0
            n_iter += 1

        return best_solution

    def solve(self) -> VRPSolution:
        best_solution = self.generate_initial_routes()
        best_objective = best_solution.get_objective_value()

        initial_temperature = (
            best_solution.get_objective_value(penalize_temp_route=False) / 20
        )
        max_iter = 10000
        max_non_improvement = 20
        n_non_improvement = 0
        temperature = initial_temperature
        alpha = 0.92

        neighbor = lambda x: x.neighbor_portfolio()

        while True:
            next_solution = self.local_search(
                best_solution, neighbor, temperature, max_iter
            )
            next_objective = next_solution.get_objective_value()
            if next_objective < best_objective:
                n_non_improvement = 0
                best_solution = next_solution
                best_objective = next_objective
                # print(best_objective, temperature)
            else:
                n_non_improvement += 1
                temperature = temperature * alpha

            if n_non_improvement >= max_non_improvement:
                break

        return best_solution


parser = ArgumentParser()
parser.add_argument("-i", type=str, required=True)


def main(args):
    input_file = Path(args.i)
    instance = VRPInstance(input_file)
    start = time.time()
    solution = instance.solve()
    # solution.export(f"{input_file.name}.sol")
    # print(solution)
    end = time.time()

    print(
        f'{{"Instance": "{input_file.name}", "Time": {end - start:.2f}, "Result": {solution.get_objective_value():.1f}, "Solution": "{solution.as_solution_str()}"}}'
    )


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
