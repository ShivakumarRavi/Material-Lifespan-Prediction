import sys
import pandas as pd
from src.custom_exception import CustomException
from src.utils import load_object
import os


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        ComponentType: str,
        StructureType: str,
        CoolRate: int,
        QuenchDuration: int,
        ForgeDuration: int,
        HeatProcessTime: int,
        NickelComposition: int,
        IronComposition: int,
        CobaltComposition: int,
        ChromiumComposition: int,
        MinorDefects: int,
        MajorDefects: int,
        EdgeDefects: int,
        InitialPosition: str,
        FormationMethod: str,
    ):
        self.ComponentType = ComponentType
        self.StructureType = StructureType
        self.CoolRate = CoolRate
        self.QuenchDuration = QuenchDuration
        self.ForgeDuration = ForgeDuration
        self.HeatProcessTime = HeatProcessTime
        self.NickelComposition = NickelComposition
        self.IronComposition = IronComposition
        self.CobaltComposition = CobaltComposition
        self.ChromiumComposition = ChromiumComposition
        self.MinorDefects = MinorDefects
        self.MajorDefects = MajorDefects
        self.EdgeDefects = EdgeDefects
        self.InitialPosition = InitialPosition
        self.FormationMethod = FormationMethod

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "ComponentType": [self.ComponentType],
                "StructureType": [self.StructureType],
                "CoolRate": [self.CoolRate],
                "QuenchDuration": [self.QuenchDuration],
                "ForgeDuration": [self.ForgeDuration],
                "HeatProcessTime": [self.HeatProcessTime],
                "NickelComposition": [self.NickelComposition],
                "IronComposition": [self.IronComposition],
                "CobaltComposition": [self.CobaltComposition],
                "ChromiumComposition": [self.ChromiumComposition],
                "MinorDefects": [self.MinorDefects],
                "MajorDefects": [self.MajorDefects],
                "EdgeDefects": [self.EdgeDefects],
                "InitialPosition": [self.InitialPosition],
                "FormationMethod": [self.FormationMethod],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as exp:
            raise CustomException(exp, sys)
