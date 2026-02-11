import logging
import shutil
from pathlib import Path
from time import strftime
from typing import Any, Callable, Dict

from wcwidth import wcswidth


def strip(msg: str) -> str:
    return msg.replace('\n', '').replace('\r', '')

class Logging:
    def __init__(
        self, 
        log_path: Path, 
        log_name: str | None = None, 
        defined: Dict[str, str] = dict(), 
        write_to_file: bool = True
    ):
        self.logger = None
        if write_to_file:
            log_path.mkdir(parents=True, exist_ok=True)
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[logging.FileHandler(log_path / f'{log_name}-{strftime("%Y_%m_%d__%H_%M_%S")}.log')],
                force=True
            )

            self.logger = logging.getLogger(log_name)

        self._defined = defined
        self.write_to_file = write_to_file
    
    def require_defined(self, name: str) -> str:
        if name not in self._defined:
            raise AttributeError(f"'{name}' is not defined.")
        return self._defined[name]
    
    def log_fn(self, log_type: str):
        return getattr(self.logger, log_type, print)
        
    def debug(self, msg: str) -> None:
        self.log_fn('debug')(msg)

    def info(self, msg: str) -> None:
        self.log_fn('info')(msg)

    def warning(self, msg: str) -> None:
        self.log_fn('warning')(msg)

    def error(self, msg: str) -> None:
        self.log_fn('error')(msg)

    def critical(self, msg: str) -> None:
        self.log_fn('critical')(msg)

    def write(self, msg: str, end: str) -> None:
        stripped = strip(msg)
        print(msg, end=end, flush=True)
        if self.write_to_file and stripped != '': self.info(stripped)

    def new_line(self):
        self.write('', end='\n')

    def __getattr__(self, name: str) -> Callable:
        sentinel = object() # private marker that can't clash with user input

        def dynamic_method(*args: Any, end: str | object = sentinel) -> None:
            msg = self.require_defined(name)
            msg = msg.format(*args)

            if end is sentinel:
                # current terminal width; fallback to 80 if stdout isn't a TTY
                width = shutil.get_terminal_size(fallback=(80, 24)).columns

                # we only care about the *visible* length of the last line
                last_line_len = wcswidth(msg.splitlines()[-1])
                
                padding = max(width - last_line_len, 0)
                end = " " * padding

            self.write(msg, end) # type: ignore

        return dynamic_method