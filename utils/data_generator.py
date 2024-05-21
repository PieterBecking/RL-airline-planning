import numpy as np
import pandas as pd
from config import config  # Ensure this import matches the structure of your project

def generate_simple_demand_data(num_airports, demand_scale, price_range, distance_scale):
    """
    Generates simple demand, price, and symmetric distance matrices.
    """
    
    demand_matrix = np.random.choice([0, 0.5*demand_scale, demand_scale], size=(num_airports, num_airports)) # demand matrix only has values of 0, 0.5*demand_scale, or demand_scale
    price_matrix = np.full((num_airports, num_airports), price_range[0]) # only one price for all flights

    # Generate a symmetric distance matrix. every distance is distance_scale
    lower_triangle = np.tril(np.full((num_airports, num_airports), distance_scale), -1)
    distance_matrix = lower_triangle + lower_triangle.T  # Create a symmetric matrix by adding the transpose of the lower triangle

    np.fill_diagonal(demand_matrix, 0)  # No demand for flights from an airport to itself
    np.fill_diagonal(price_matrix, 0)   # No ticket price needed for non-existent flights
    np.fill_diagonal(distance_matrix, 0)  # No distance from an airport to itself

    return demand_matrix, price_matrix, distance_matrix

def generate_realistic_demand_data(num_airports, demand_scale, price_range, distance_scale):
    """
    Generates realistic demand, price, and symmetric distance matrices.
    """
    # For the purpose of this example, this function will generate the same kind of data as the simple version
    # Realistic data generation logic can be implemented here
    demand_matrix = np.random.normal(loc=demand_scale, scale=demand_scale/5, size=(num_airports, num_airports)).round().clip(min=0)
    price_matrix = np.random.uniform(low=price_range[0], high=price_range[1], size=(num_airports, num_airports)).round()

    # Generate a symmetric distance matrix
    lower_triangle = np.tril(np.random.randint(100, distance_scale, size=(num_airports, num_airports)), -1)
    distance_matrix = lower_triangle + lower_triangle.T  # Create a symmetric matrix by adding the transpose of the lower triangle

    np.fill_diagonal(demand_matrix, 0)  # No demand for flights from an airport to itself
    np.fill_diagonal(price_matrix, 0)   # No ticket price needed for non-existent flights
    np.fill_diagonal(distance_matrix, 0)  # No distance from an airport to itself

    return demand_matrix, price_matrix, distance_matrix

def generate_demand_data():
    """
    Generates and saves demand, price, and symmetric distance matrices to CSV files based on the configuration.
    """
    num_airports = config['num_airports']
    demand_scale = config['demand_scale']
    price_range = config['price_range']
    distance_scale = config['distance_scale']

    if config['data_generator'] == 'simple':
        demand_matrix, price_matrix, distance_matrix = generate_simple_demand_data(num_airports, demand_scale, price_range, distance_scale)
    elif config['data_generator'] == 'realistic':
        demand_matrix, price_matrix, distance_matrix = generate_realistic_demand_data(num_airports, demand_scale, price_range, distance_scale)
    else:
        raise ValueError("Invalid data_generator type specified in config")

    # Save to CSV
    pd.DataFrame(demand_matrix).to_csv('data/demand_matrix.csv', index=False)
    pd.DataFrame(price_matrix).to_csv('data/price_matrix.csv', index=False)
    pd.DataFrame(distance_matrix).to_csv('data/distance_matrix.csv', index=False)

# Run the function to generate data
if __name__ == "__main__":
    generate_demand_data()  # Uses settings from config.py
