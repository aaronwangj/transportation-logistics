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

    TEMPORARY_VEHICLE_PENALTY = 1e6

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

    def get_objective_value(self) -> float:
        objective_value = 0.0
        for i, route in enumerate(self.routes):
            if (
                i != len(self.routes) - 1
                and self.get_route_capacity(route) > self.config.vehicle_capacity
            ):
                return float("inf")
            distance = self.get_route_distance(route)
            if i == len(self.routes) - 1:
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


class VRPInstance:
    def __init__(self, filename: str):
        self.config = VRPInstanceConfig.from_file(filename)
        print(self.config)

    def generate_initial_routes(self) -> VRPSolution:
        is_taken = [False for _ in range(self.config.n_customers)]
        is_taken[0] = True
        thetas = [
            math.atan2(self.config.ys[i], self.config.xs[i])
            for i in range(self.config.n_customers)
        ]
        sorted_customers = sorted(
            range(1, self.config.n_customers), key=lambda i: thetas[i]
        )
        index_to_sorted_index = {
            i: sorted_i for sorted_i, i in enumerate(sorted_customers)
        }

        routes = []

        for i in range(self.config.n_vehicles):
            # random initial point
            available_customers = [
                i for i, is_taken in enumerate(is_taken) if not is_taken
            ]
            if len(available_customers) == 0:
                break
            total_demand = 0
            route = []
            next_customer = random.choice(available_customers)
            while True:
                if (
                    total_demand + self.config.demands[next_customer]
                    > self.config.vehicle_capacity
                ):
                    break

                total_demand += self.config.demands[next_customer]
                route.append(next_customer)
                is_taken[next_customer] = True
                for i in range(1, self.config.n_customers):
                    option = sorted_customers[
                        (index_to_sorted_index[next_customer] + i)
                        % (self.config.n_customers - 1)
                    ]
                    if not is_taken[option]:
                        next_customer = option
                        break
                if route[-1] == next_customer:
                    break
            routes.append(route)
        # fill up with random routes
        for _ in range(self.config.n_vehicles - len(routes)):
            routes.append([])
        # put the remaining unassigned customers in the temporary vehicle
        routes.append([i for i, is_taken in enumerate(is_taken) if not is_taken])
        return VRPSolution(routes, self.config)

    def solve(self) -> VRPSolution:
        solution = self.generate_initial_routes()
        return self.local_search(
            solution,
        )

    def local_search(
        self,
        initial_solution: VRPSolution,
        neighbor: Callable[[VRPSolution], VRPSolution],
        timeout: float,
        non_improving_limit: int = 4,
    ) -> VRPSolution:
        start = time.time()

        n_non_improving = 0
        best_solution = initial_solution
        best_objective = initial_solution.get_objective_value()
        while time.time() - start < timeout and n_non_improving < non_improving_limit:
            next_solution = neighbor()
            next_objective = next_solution.get_objective_value()
            if next_objective < best_objective:
                best_solution = next_solution
                best_objective = next_objective
                n_non_improving = 0
            else:
                n_non_improving += 1

        return best_solution


parser = ArgumentParser()
parser.add_argument("-i", type=str, required=True)


def main(args):
    input_file = Path(args.i)
    instance = VRPInstance(input_file)
    start = time.time()
    solution = instance.solve()
    end = time.time()

    print(
        f'{{"Instance": "{input_file.name}", "Time": {end - start:.2f}, "Result": {solution.get_objective_value():.1f}, "Solution": "{solution.as_solution_str()}"}}'
    )


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
