from setuptools import setup

package_name = "driverless_perception_ros"

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
    description="ROS 2 nodes wrapping the driverless_perception pipelines (lidar, mono, stereo).",
    license="MIT",
    entry_points={"console_scripts": ["mono_node = driverless_perception_ros.mono_node:main",
                                      "stereo_node = driverless_perception_ros.stereo_node:main"]},
)
