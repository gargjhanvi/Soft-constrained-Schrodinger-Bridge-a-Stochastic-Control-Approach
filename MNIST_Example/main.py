import subprocess
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--beta", type=str, help="Regularization parameter in soft contrained Schrodinger bridge problem")
args = parser.parse_args()
beta_value = args.beta
print(f"Value of 'beta': {beta_value}")
py_files_to_run = [
    ('density_ratio.py',{}),
    ('score_ref.py', {'beta': beta_value}),
    ('score_obj.py', {'beta': beta_value}),
    ('runner.py', {'beta': beta_value})
]
for py_file, variables in py_files_to_run:
    try:
        command = ['python3', py_file]

        for variable, value in variables.items():
            command.extend([f'--{variable}', value])
          
        subprocess.run(command)
        print(f'Successfully executed {py_file}')
    except Exception as e:
        print(f'Error executing {py_file}: {e}')
