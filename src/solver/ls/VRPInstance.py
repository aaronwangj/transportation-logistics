import numpy as np
import time
import sys, json

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
        self.numCustomers = -float('inf')
        self.numVehicles = -float('inf')
        self.vehicleCapacity = -float('inf')
        self.demandOfCustomer = []
        self.xCoordOfCustomer = []
        self.yCoordOfCustomer = []
        self.customers = []

        with open(self.filePath, 'r') as file:
            self.rawData = file.read().split('\n')

        self.rawData = [[float(x) for x in y.split(' ') if x] for y in self.rawData if y]
        self.numCustomers = self.rawData[0][0]
        self.numVehicles = self.rawData[0][1]
        self.vehicleCapacity = self.rawData[0][2]
        print('CUSTOMERS: ', int(self.numCustomers))
        print('VEHICLES: ', int(self.numVehicles))
        print('CAPACITY: ', self.vehicleCapacity)
        print('RAW: ', self.rawData)

        for index, customer in enumerate(self.rawData[2:]):
            if not customer:
                continue
            self.demandOfCustomer.append(customer[0])
            self.xCoordOfCustomer.append(customer[1])
            self.yCoordOfCustomer.append(customer[2])
            self.customers.append(Customer(index+1, customer[0], customer[1], customer[2]))
        print('DEMANDS: ', self.demandOfCustomer)
        print('X COORDS: ', self.xCoordOfCustomer)
        print('Y COORDS: ', self.yCoordOfCustomer)

    def initialize(self):
        pass

    def solve(self):
        pass

    def calcDistance(self, solution):
        totalDistance = 0

        for vehicle in solution:
            curDistance = 0
            for i in range(len(vehicle)):
                if i == 0:
                    curDistance += self.euclideanDistance((0,0), (vehicle[i].x, vehicle[i].y))
                else:
                    curDistance += self.euclideanDistance((vehicle[i-1].x, vehicle[i-1].y), (vehicle[i].x, vehicle[i].y))

            totalDistance += curDistance
        return totalDistance

    def euclideanDistance(self, point1, point2):
        return ((point1[0]-point2[0])**2 + (point1[1]-point2[0])**2)**.5


    def output(self):
        result = {}
        result['Instance'] = self.filePath[:-4] if self.filePath[-4:] == '.vrp' else self.filePath
        start = time.time()
        output = self.solve()
        end = time.time()
        result['Time'] = '%.2f' % (end-start)
        result['Result'] = 'N/A'
        result['Solution'] = output
        return result

def main():
    filePath = sys.argv[1]
    solver = VRPInstance(filePath)
    result = solver.output()
    print(json.dumps(result))

if __name__ == "__main__":
    main()