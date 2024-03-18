from datetime import datetime, timedelta
import numpy as np

def generate_date_array(n, start_date='2024-01-01'):
    """
    Generate an array of dates starting from the given start_date, incrementing by one day for n elements.

    Args:
    - start_date: A string representing the start date in the format 'YYYY-MM-DD'.
    - n: An integer representing the number of elements in the array.

    Returns:
    - An array of datetime objects representing dates.
    """
    # Convert the start_date string to a datetime object
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    
    # Initialize an empty list to store the generated dates
    date_array = []
    
    # Generate dates by incrementing start_date by one day for n elements
    for i in range(n):
        # Append the date in the desired format to the date_array
        date_array.append(start_date.strftime('%Y-%m-%d'))
        start_date += timedelta(days=1)
    
    return np.array(date_array)