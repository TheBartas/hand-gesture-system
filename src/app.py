from gesture_console import GestureConsole
from ModuleM.gesture_manager import GestureManager
from ModuleD import StaticGestureDetector, DynamicGestureDetector, GestureVisualization

if __name__ == "__main__":

    # TODO: add commands for visualization and server 

    models = {
        1: GestureManager.import_static(file_name="static"),
        2: GestureManager.import_dynamic(file_name="dynamic")
    }

    console = GestureConsole(models)
    console.run()