# Notas de Configuración y Ejecución Local

Este documento registra los ajustes realizados en el proyecto para permitir su correcta ejecución en entornos locales (Windows / CPU o sin GPU dedicada exclusiva) y el comando directo para correr el agente.

---

## 1. Comando de Ejecución Rápida

Para correr el agente con el entorno virtual local (`.venv`), puedes usar el archivo batch creado en la raíz:

```cmd
run_local.bat
```

O ejecutarlo directamente en PowerShell / CMD:

```powershell
.\.venv\Scripts\python.exe steve1\run_agent\run_agent.py --custom_text_prompt "tbag" --save_dirpath data\generated_videos\test --gameplay_length 100
```

### Parámetros comunes:
- `--custom_text_prompt`: El objetivo o indicación en texto para el agente (ej. `"tbag"`, `"mine dirt"`, etc.).
- `--save_dirpath`: Directorio donde se guardarán los videos generados del gameplay.
- `--gameplay_length`: Número de pasos (steps) que durará la sesión (ej. `100`).
- `--prior_weights`: (Opcional) Ruta al archivo de pesos del VAE previo.

---

## 2. Modificaciones realizadas en el código

Para asegurar compatibilidad y evitar errores en máquinas locales donde CUDA no esté presente o no soporte `amp.autocast`:

1. **`steve1/run_agent/run_agent.py`**:
   - Se sustituyó la llamada estricta `with torch.cuda.amp.autocast():` por un contexto dinámico (`autocast_ctx = torch.cuda.amp.autocast if torch.cuda.is_available() else contextlib.nullcontext`).
   - Se añadió soporte para sobrescribir los pesos del prior (`prior_weights`) en los argumentos de ejecución.

2. **`steve1/utils/embed_utils.py`**:
   - En `get_prior_embed`, se adaptó el autocast para usar `contextlib.nullcontext` cuando CUDA no esté disponible.

3. **`steve1/utils/mineclip_agent_env_utils.py`**:
   - En `make_agent`, se cambió el parámetro hardcodeado `device='cuda'` por la variable global configurable `DEVICE`.

4. **`.gitignore`**:
   - Se añadió exclusión para `*.pt` evitando subir checkpoints y pesos pesados al repositorio.
