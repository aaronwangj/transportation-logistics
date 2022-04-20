import json
import sys
import time

import numpy as np
from sklearn.metrics import euclidean_distances


class Customer:
    def __init__(self, index, demand, x, y):
        self.x = x
        self.y = y
        self.demand = demand
        self.index = index


class VRPInstance:
    def __init__(self, filePath):
        self.filePath = filePath
        self.rawData = []
        self.numCustomers = -float("inf")
        self.numVehicles = -float("inf")
        self.vehicleCapacity = -float("inf")
        self.demandOfCustomer = []
        self.xCoordOfCustomer = []
        self.yCoordOfCustomer = []
        self.customers = []

        with open(self.filePath, "r") as file:
            self.rawData = file.read().split("\n")

        self.rawData = [
            [float(x) for x in y.split(" ") if x] for y in self.rawData if y
        ]
        self.numCustomers = int(self.rawData[0][0]) - 1
        self.numVehicles = int(self.rawData[0][1])
        self.vehicleCapacity = int(self.rawData[0][2])

        for index, customer in enumerate(self.rawData[2:]):
            if not customer:
                continue
            self.demandOfCustomer.append(customer[0])
            self.xCoordOfCustomer.append(customer[1])
            self.yCoordOfCustomer.append(customer[2])
            self.customers.append(
                Customer(index, customer[0], customer[1], customer[2])
            )

    def initialize(self):
        is_taken = [False for _ in range(self.numCustomers)]
        routes = []
        throttle = (
            self.vehicleCapacity
        )  # np.sum(self.demandOfCustomer) / self.numVehicles
        print(f"Throttle: {throttle} | Capacity {self.vehicleCapacity}")
        thetas = [np.arctan2(c.y, c.x) for c in self.customers]
        sorted_customers = sorted(range(self.numCustomers), key=lambda i: thetas[i])
        index_to_sorted_index = {
            i: sorted_i for sorted_i, i in enumerate(sorted_customers)
        }

        for i in range(self.numVehicles):
            # random initial point
            available_customers = [
                i for i, is_taken in enumerate(is_taken) if not is_taken
            ]
            if len(available_customers) == 0:
                break
            total_demand = 0
            route = []
            next_customer = np.random.choice(available_customers)
            while True:
                if (
                    total_demand + self.demandOfCustomer[next_customer]
                    > self.vehicleCapacity
                ):
                    break

                total_demand += self.demandOfCustomer[next_customer]
                route.append(next_customer)
                is_taken[next_customer] = True
                for i in range(1, self.numCustomers):
                    option = sorted_customers[
                        (index_to_sorted_index[next_customer] + i) % self.numCustomers
                    ]
                    if not is_taken[option]:
                        next_customer = option
                        break
                if route[-1] == next_customer:
                    break
                if total_demand > throttle:
                    break
            routes.append(route)
        # fill up with random routes
        for _ in range(self.numVehicles - len(routes)):
            routes.append([])
        return routes

    def solve(self):
        while True:
            is_ok = True
            routes = self.initialize()
            is_hit = [False for _ in range(self.numCustomers)]
            for route in routes:
                for point in route:
                    if is_hit[point]:
                        is_ok = False
                        break
                    is_hit[point] = True
            for i in range(self.numCustomers):
                if not is_hit[i]:
                    is_ok = False
                    break
            if is_ok:
                break

    def calcDistance(self, solution):
        totalDistance = 0

        for vehicle in solution:
            curDistance = 0
            for i in range(len(vehicle)):
                if i == 0:
                    curDistance += self.euclideanDistance(
                        (0, 0), (vehicle[i].x, vehicle[i].y)
                    )
                else:
                    curDistance += self.euclideanDistance(
                        (vehicle[i - 1].x, vehicle[i - 1].y),
                        (vehicle[i].x, vehicle[i].y),
                    )

            totalDistance += curDistance
        return totalDistance

    def euclideanDistance(self, point1, point2):
        return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[0]) ** 2) ** 0.5

    def output(self):
        result = {}
        result["Instance"] = (
            self.filePath[:-4] if self.filePath[-4:] == ".vrp" else self.filePath
        )
        start = time.time()
        output = self.solve()
        end = time.time()
        result["Time"] = "%.2f" % (end - start)
        result["Result"] = "N/A"
        result["Solution"] = output
        return result


def main():
    filePath = sys.argv[1]
    solver = VRPInstance(filePath)
    result = solver.output()
    print(json.dumps(result))


if __name__ == "__main__":
    main()
