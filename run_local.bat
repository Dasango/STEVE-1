@echo off
.\.venv\Scripts\python.exe steve1\run_agent\run_agent.py --custom_text_prompt "tbag" --save_dirpath data\generated_videos\test --gameplay_length 100 %*
