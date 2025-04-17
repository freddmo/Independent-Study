import pickle

def read_first_four_iterations_from_pkl(file_path):
    """
    Reads the data for the first four iterations from a .pkl file and returns them as a list.
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        if isinstance(data, list):
            return data[:4]  # Return the first four elements
        elif isinstance(data, dict):
            iterations = []
            for i in range(4):
                if i in data:
                    iterations.append(data[i])
            return iterations
        elif hasattr(data, '__iter__'):
            iterations = []
            iterator = iter(data)
            for _ in range(4):
                try:
                    iterations.append(next(iterator))
                except StopIteration:
                    break
            return iterations
        else:
            return [data]  # Return the single item as a list

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error reading pkl file: {e}")
        return None

def print_beginning(data, num_items=10):
    """
    Prints the beginning of a data structure.
    """
    if data is None:
        print("No data to print.")
        return

    print("Beginning of the data:")
    if isinstance(data, list):
        for i, item in enumerate(data):
            print(f"Iteration {i+1}:")
            if isinstance(item, list):
                print(f"  {item[:num_items]}")
            elif isinstance(item, dict):
                keys = list(item.keys())[:num_items]
                for key in keys:
                    print(f"  {key}: {item[key]}")
            elif isinstance(item, set):
                print(f"  {list(item)[:num_items]}")
            elif isinstance(item, tuple):
                print(f"  {item[:num_items]}")
            elif hasattr(item, '__iter__'):
                try:
                    iterator = iter(item)
                    print("  ", end="")
                    for _ in range(num_items):
                        print(next(iterator), end=", ")
                    print("...")
                except StopIteration:
                    print("  (empty iterable)")
            else:
                print(f"  {item}")
            if i < len(data) - 1:
                print("-" * 20)
    elif isinstance(data, dict):
        keys = list(data.keys())[:num_items]
        for key in keys:
            print(f"  {key}: {data[key]}")
    elif isinstance(data, set):
        print(list(data)[:num_items])
    elif isinstance(data, tuple):
        print(data[:num_items])
    elif hasattr(data, '__iter__'):
        try:
            iterator = iter(data)
            for _ in range(num_items):
                print(next(iterator))
        except StopIteration:
            pass
    else:
        print(data)  # If it's a single item

# Example usage:
file_path = r'C:\Users\SAC\Documents\Freddy_Folder\FAU\Spring 2025\Independent Study\vrp_nazari100_validation_seed4321-lkh.pkl'
first_four_iterations_data = read_first_four_iterations_from_pkl(file_path)

if first_four_iterations_data is not None:
    print("Data from the first four iterations:")
    print_beginning(first_four_iterations_data)
else:
    print("Could not read the first four iterations data.")

