import importlib.util

def load_user_dataset(dataset_path="user_dataset.py"):
    """Dynamically loads the user's dataset."""
    spec = importlib.util.spec_from_file_location("user_dataset", dataset_path)
    user_dataset = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(user_dataset)
    return user_dataset.get_dataloader()  # Calls a function in user_dataset.py

# Load user dataset
dataloader = load_user_dataset()
