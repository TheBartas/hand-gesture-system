from ModuleM.dynamic_model import DynamicGestureModel
from ModuleM.static_model import StaticGestureModel
from ModuleM.paths import MODEL_JSON_PATH
import json

class GestureManager:

    @staticmethod
    def import_models() -> list:
        files = MODEL_JSON_PATH.rglob('*.json')
        data = []
        for file in files:
            with open(file, 'r') as f:
                model = json.load(f)

                if model["class"] == "static":
                    data.append(StaticGestureModel.from_json(model=model))
                elif model["class"] == "dynamic":
                    data.append(DynamicGestureModel.from_json(model=model))
                else:
                    raise ValueError("[Error] Unknown <class>")
        return data

    @staticmethod
    def import_static_list() -> list :
        files = MODEL_JSON_PATH.rglob('*.json')
        data = []
        for file in files:
            with open(file, 'r') as f:
                model = json.load(f)

                if model["class"] == "static":
                    data.append(StaticGestureModel.from_json(model=model))
        return data
    
    @staticmethod
    def import_static(file_name : str) -> StaticGestureModel :
        with open(MODEL_JSON_PATH / f'{file_name}.json', 'r') as f:
            model = json.load(f)

        if model["class"] != "static":
            raise ValueError("[Error] Unknown <class>")

        return StaticGestureModel.from_json(model=model)

    @staticmethod
    def import_dynamic_list() -> list :
        files = MODEL_JSON_PATH.rglob('*.json')
        data = []
        for file in files:
            with open(file, 'r') as f:
                model = json.load(f)

                if model["class"] == "dynamic":
                    data.append(DynamicGestureModel.from_json(model=model))
        return data

    @staticmethod
    def import_dynamic(file_name : str) -> DynamicGestureModel :
        with open(MODEL_JSON_PATH / f'{file_name}.json', 'r') as f:
            model = json.load(f)

        if model["class"] != "dynamic":
            raise ValueError("[Error] Unknown <class>")

        return DynamicGestureModel.from_json(model=model)
