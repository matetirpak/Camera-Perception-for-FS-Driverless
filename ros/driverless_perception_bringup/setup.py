from glob import glob

from setuptools import setup

package_name = "driverless_perception_bringup"

setup(
    name=package_name,
    version="0.1.0",
    packages=[],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
        ("share/" + package_name + "/config", glob("config/*")),
        ("share/" + package_name + "/foxglove", glob("foxglove/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Máté Tirpák",
    maintainer_email="mate.tirpak@gmail.com",
    description="Launch files, parameters and the Foxglove layout for the perception stack.",
    license="MIT",
)
