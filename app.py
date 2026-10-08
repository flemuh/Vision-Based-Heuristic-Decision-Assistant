from vision_engine.dpi import enable_dpi_awareness

enable_dpi_awareness()

from pathlib import Path
import multiprocessing as mp

from vision_engine.gui import run


if __name__ == "__main__":
    mp.freeze_support()
    run(Path(__file__).resolve().parent)
