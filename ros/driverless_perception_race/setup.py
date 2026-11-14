from setuptools import setup

package_name = "driverless_perception_race"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Máté Tirpák",
    maintainer_email="mate.tirpak@gmail.com",
    description="Ground truth race harness: GT transform and markers, oracle driver, race supervisor, recording helpers.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "ground_truth_node = driverless_perception_race.ground_truth_node:main",
            "oracle_driver_node = driverless_perception_race.oracle_driver_node:main",
            "race_supervisor_node = driverless_perception_race.race_supervisor_node:main",
            "image_compressor_node = driverless_perception_race.image_compressor_node:main",
        ],
    },
)
