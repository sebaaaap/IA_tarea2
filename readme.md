# Crear entorno virtual

Instrucciones para crear y usar un entorno virtual en Linux (bash). Incluye instalar dependencias y ejecutar el notebook del proyecto.

1) Crear el entorno virtual

```bash
python3 -m venv IA_tarea2
```

2) Activar el entorno (bash)

```bash
source IA_tarea2/bin/activate
```

3) (Opcional) Actualizar pip dentro del entorno

```bash
python -m pip install --upgrade pip
```

4) Instalar dependencias

```bash
pip install -r requirements.txt
```

Si no 

```bash
pip install pandas numpy scikit-learn openpyxl jupyter
```

5) Ejecutar el notebook del proyecto

```bash
jupyter notebook script_parte1.ipynb
```
aqui colocar "run all cells"

6) Desactivar el entorno cuando termines

```bash
deactivate
```

Notas adicionales
- Si al crear el venv obtienes un error, instala la dependencia del sistema (Ubuntu/Debian):

```bash
sudo apt update
sudo apt install python3-venv
```


