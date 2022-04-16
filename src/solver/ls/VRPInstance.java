package solver.ls;

import java.io.File;
import java.io.FileNotFoundException;

import java.util.Scanner;

public class VRPInstance
{
  // VRP Input Parameters
  int numCustomers;        		// the number of customers	   
  int numVehicles;           	// the number of vehicles
  int vehicleCapacity;			// the capacity of the vehicles
  int[] demandOfCustomer;		// the demand of each customer
  double[] xCoordOfCustomer;	// the x coordinate of each customer
  double[] yCoordOfCustomer;	// the y coordinate of each customer
  
  public VRPInstance(String fileName)
  {
    Scanner read = null;
    try
    {
      read = new Scanner(new File(fileName));
    } catch (FileNotFoundException e)
    {
      System.out.println("Error: in VRPInstance() " + fileName + "\n" + e.getMessage());
      System.exit(-1);
    }

    numCustomers = read.nextInt(); 
    numVehicles = read.nextInt();
    vehicleCapacity = read.nextInt();
    
    System.out.println("Number of customers: " + numCustomers);
    System.out.println("Number of vehicles: " + numVehicles);
    System.out.println("Vehicle capacity: " + vehicleCapacity);
      
    demandOfCustomer = new int[numCustomers];
	xCoordOfCustomer = new double[numCustomers];
	yCoordOfCustomer = new double[numCustomers];
	
    for (int i = 0; i < numCustomers; i++)
	{
		demandOfCustomer[i] = read.nextInt();
		xCoordOfCustomer[i] = read.nextDouble();
		yCoordOfCustomer[i] = read.nextDouble();
	}
	
	for (int i = 0; i < numCustomers; i++)
		System.out.println(demandOfCustomer[i] + " " + xCoordOfCustomer[i] + " " + yCoordOfCustomer[i]);
  }
}
