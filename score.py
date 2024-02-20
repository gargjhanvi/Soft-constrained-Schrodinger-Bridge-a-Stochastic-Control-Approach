import subprocess
import argparse

# Create an ArgumentParser object
parser = argparse.ArgumentParser()

# Define the command-line argument
parser.add_argument("--beta", type=str, help="Regularization parameter in soft contrained Schrodinger bridge problem")

# Parse the command-line arguments
args = parser.parse_args()

# Access the value of the '--var' argument
beta_value = args.beta

# Your script can now use 'var_value' as needed
print(f"Value of 'beta': {beta_value}")

# List of .py files to trigger along with variable names and values
py_files_to_run = [
    ('score_ref.py', {'beta': beta_value}),
    ('score_obj.py', {'beta': beta_value})

]

#Loop through the list and run each .py file with its variables
for py_file, variables in py_files_to_run:
    try:
        # Construct the command to run the .py file with variables
        command = ['python3', py_file]

        # Append variables and their values to the command
        for variable, value in variables.items():
            command.extend([f'--{variable}', value])
          
        subprocess.run(command)
        print(f'Successfully executed {py_file}')
    except Exception as e:
        print(f'Error executing {py_file}: {e}')
