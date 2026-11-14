from pathlib import Path

import pytest

FSDS_REPO = Path.home() / "Formula-Student-Driverless-Simulator"


@pytest.fixture
def droneport_csv() -> Path:
    p = FSDS_REPO / "maps" / "FormulaElectricBelgium" / "track_droneport.csv"
    if not p.exists():
        pytest.skip("FSDS repo not found")
    return p


@pytest.fixture
def default_settings() -> dict:
    return {"Vehicles": {"FSCar": {
        "Sensors": {"Lidar1": {"SensorType": 6, "X": 0.45, "Y": 0, "Z": 0.55, "NumberOfLasers": 4,
                               "PointsPerScan": 4096, "VerticalFOVUpper": 5, "VerticalFOVLower": -5,
                               "HorizontalFOVStart": -90, "HorizontalFOVEnd": 90, "RotationsPerSecond": 10}},
        "Cameras": {"cam1": {"CaptureSettings": [{"ImageType": 0, "Width": 785, "Height": 785, "FOV_Degrees": 90}],
                             "X": -0.3, "Y": -0.16, "Z": 0.8}}}}}
