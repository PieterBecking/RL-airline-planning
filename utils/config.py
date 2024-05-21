config = {
    "num_airports": 5,  # Number of airports
    "time_slots": 24,   # Number of time slots (assuming each slot is an hour)
    "demand_scale": 100, # Scale for demand generation
    "price_range": (100, 300), # Range for ticket prices
    "distance_scale": 1000, # Maximum distance scale for flights
    "aircraft_capacity": 180,  # Passenger capacity of the aircraft
    "cost_parameter": 10,  # Cost per unit distance flown
    "max_aircraft": 1,  # Maximum number of aircraft available
    "data_generator": "simple",  # Options: "simple", "realistic" 
    "agent": "dqn_keras"  # Options: "q_learning", "dqn", "ppo", "dqn_keras"
}
