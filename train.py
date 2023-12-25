import subprocess

# Número de veces que quieres ejecutar el script
n = 100

# Ruta al script que quieres ejecutar
script_path = 'snakeoil3_gym.py'

for i in range(n):
    print("-----------------------------------------")
    print(f'Ejecutando {script_path}, ejecución número {i+1}')
    print("-----------------------------------------")
    
    # Ejecutar el script y esperar a que termine
    subprocess.run(['python', script_path], check=True)
    
    print("-----------------------------------------")
    print(f'Terminada ejecución número {i+1}')
    print("-----------------------------------------")
