from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.shortcuts import choice


class GestureConsole:
    def __init__(self, models={}, detectors={}, visualization={}, commands=None) :
        self._models = models
        self._curr_model = None

        self._commands = commands or [
            "sel-mod",          # Select a model 
            "li-mod",           # Lists models 
            "curr-mod",         # Current model (shows current (chosen) model)
            "f-init",           # Files initialize 
            "cll-data",         # Collect data
            "bdl-mod",          # Build model
            "sv-mod",           # Model save
            "cnvt-mod",         # Convert model to tflite
            "mod-i",            # Show model info
            "q"                 # Quit
        ]

        self._command_map = {
            self._commands[3]: ("Files initialize", "init_data_files"),
            self._commands[4]: ("Collect data", "collect_data"),
            self._commands[5]: ("Build model", "build_model"),
            self._commands[6]: ("Save model", "save"),
            self._commands[7]: ("Convert model to tflite", "convert_to_tflite"),
            self._commands[8]: ("Show model info", "info"),
        }

        self._options = [(value, f'{value}') for value in self._models.values()]

        self.session = PromptSession(
            completer=WordCompleter(self._commands, ignore_case=True)
        )

    def run(self):
        print("[info] Console active. Type 'q' to quit.")
        while True:
            try:
                opt = self.session.prompt(">>> ").strip()
                if not self.__run_command(opt):
                    break
            except KeyboardInterrupt:
                continue     # Ctrl+C 
            except EOFError: # Ctrl+D 
                print("Exiting...")
                break

    def __run_command(self, cmd) :

        if cmd == self._commands[0]: 
            self._curr_model = choice(
                message="Choose a model as current:",
                options=self._options,
                default=self._models[1])
            
        elif cmd == self._commands[1]:
            for key, value in self._models.items():
                print(f"({key}) | {value}")

        elif cmd == self._commands[2]:
            print(self._curr_model or f'[error] No chosen model.')

        elif cmd in self._command_map:
            msg, method = self._command_map[cmd]
            print(f"[info] {msg}")

            if self._curr_model is None:
                print("[error] No chosen model.")
                return True
            
            getattr(self._curr_model, method)()

        elif cmd == self._commands[9]:
            print("[error] Exiting...")
            return False

        else:
            print(f"Unknown command: {cmd}")

        return True




