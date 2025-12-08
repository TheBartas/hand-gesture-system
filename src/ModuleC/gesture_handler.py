from ModuleC import Controller

class GestureHandler:
    def __init__(
            self, 
            controller : Controller = None,
            pan_right: int = 5,
            pan_left: int = -5,
            tilt_up: int = 5,
            tilt_down: int = -5,
            zoom_in: int = 5,
            zoom_out: int = -5,
    ) :
        self._controller=controller
        self._command_map = {
            "one_finger_left": lambda: self._controller.pan(pan_left),
            "one_finger": lambda: self._controller.pan(pan_right), # one_finger_right
            "one_finger_circle": lambda: self._controller.tilt(tilt_down),
            "one_finger_down": lambda: self._controller.tilt(tilt_up),
            "two_fingers": lambda: self._controller.zoom(zoom_in),
            "thumb_up": lambda: self._controller.zoom(zoom_out),
        }
    
    async def handle(self,  gesture : str) :
        if gesture in self._command_map:
            await self._command_map[gesture]()
            print(f"[info] Gesture executed: {gesture}")
        if gesture == 'OK':
            print("[info] OK")
