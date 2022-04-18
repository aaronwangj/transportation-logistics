import numpy as np
import time
import sys, json

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

        for customer in self.rawData[2:]:
            if not customer:
                continue
            self.demandOfCustomer.append(customer[0])
            self.xCoordOfCustomer.append(customer[1])
            self.yCoordOfCustomer.append(customer[2])
        print('DEMANDS: ', self.demandOfCustomer)
        print('X COORDS: ', self.xCoordOfCustomer)
        print('Y COORDS: ', self.yCoordOfCustomer)

    def solve(self):
        pass

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