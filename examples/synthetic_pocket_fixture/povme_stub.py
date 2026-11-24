"""POVME-format contract stub for CI; this is not the POVME package."""

from pathlib import Path
import sys


def main() -> None:
    settings: dict[str, str] = {}
    for line in Path(sys.argv[1]).read_text().splitlines():
        if " " in line:
            key, value = line.split(" ", 1)
            settings[key] = value
    frame_count = Path(settings["PDBFileName"]).read_text().count("ENDMDL")
    output = Path(settings["OutputFilenamePrefix"] + "volumes.tabbed.txt")
    output.write_text("".join(f"{index}\t{100.0 + index}\n" for index in range(frame_count)))


if __name__ == "__main__":
    main()
