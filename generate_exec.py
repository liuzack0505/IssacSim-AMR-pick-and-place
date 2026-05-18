from pathlib import Path


project_dir = Path(__file__).resolve().parent
main_path = project_dir / "main.py"
exec_path = project_dir / "exec.py"

if not main_path.is_file():
    raise FileNotFoundError(f"Cannot find main.py at {main_path}")

exec_path.write_text(
    'exec(open(r"' + str(main_path) + '",\n'
    '     encoding="utf-8").read())\n',
    encoding="utf-8",
)

print(f"Generated {exec_path} for {main_path}")
