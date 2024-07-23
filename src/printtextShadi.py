import os

# Set default path to results
RESULT_SAVE_PATH = "Results"

def printTextShadi(*args, **kwargs):
    # Check if env variable RESULT_SAVE_PATH is set; if so, overwrite the variable
    env_result_save_path = os.getenv('RESULT_SAVE_PATH')
    if env_result_save_path:
        RESULT_SAVE_PATH = env_result_save_path
    os.makedirs(RESULT_SAVE_PATH, exist_ok=True)
    with open(f'{RESULT_SAVE_PATH}/output.txt', 'a') as f:
        for arg in args:
            print(arg)
            f.write(f"{arg}\n")
        for key, value in kwargs.items():
            print(f"{key}: {value}")
            f.write(f"{key}: {value}\n")
