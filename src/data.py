from typing import Any, Dict, List
from .group import Group
from dataclasses import dataclass

@dataclass
class ObstacleData:
    ID: int
    corners: List[List[float]]

@dataclass
class VehicleData:
    



class DataSaver():
    def __init__(self, file_name: str):
        self.file_name: str = file_name
        self.data: List[Dict] = []

    def update(self, group:Group):
        """ Add new information to the data list.
        """
        data: Dict = {}
        for obstacle in group.obstacles:
            obstacle_data = ObstacleData(ID=obstacle.ID, corners=obstacle.corners)
            data["obstacles"] = obstacle_data
        
