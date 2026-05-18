from pathlib import Path
import re


project_dir = Path(__file__).resolve().parent
main_path = project_dir / "main.py"
exec_path = project_dir / "exec.py"
scene_usd_path = project_dir / "asset" / "hospital.usd"
robot_usd_path = project_dir / "asset" / "custom_robot.usd"

if not main_path.is_file():
    raise FileNotFoundError(f"Cannot find main.py at {main_path}")
if not scene_usd_path.is_file():
    raise FileNotFoundError(f"Cannot find hospital.usd at {scene_usd_path}")
if not robot_usd_path.is_file():
    raise FileNotFoundError(
        f"Cannot find custom_robot.usd at {robot_usd_path}")


def _raw_path_assignment(name, path):
    return f'{name} = r"{path}"'


main_text = main_path.read_text(encoding="utf-8")
replacements = {
    "SCENE_USD_PATH": scene_usd_path,
    "ROBOT_USD_PATH": robot_usd_path,
}

for constant_name, usd_path in replacements.items():
    pattern = rf'^{constant_name}\s*=\s*r?["\'].*?["\']\s*$'
    main_text, count = re.subn(
        pattern,
        lambda _match, name=constant_name, path=usd_path: _raw_path_assignment(
            name, path
        ),
        main_text,
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise RuntimeError(f"Could not update {constant_name} in {main_path}")

main_path.write_text(main_text, encoding="utf-8")

exec_path.write_text(
    'exec(open(r"' + str(main_path) + '",\n'
    '     encoding="utf-8").read())\n',
    encoding="utf-8",
)

print(f"Updated USD paths in {main_path}")
print(f"Generated {exec_path} for {main_path}")
