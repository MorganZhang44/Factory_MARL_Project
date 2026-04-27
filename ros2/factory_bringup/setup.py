from glob import glob

from setuptools import setup

package_name = "factory_bringup"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="yyz",
    maintainer_email="yyz@example.com",
    description="Launch files for Factory MARL ROS2 integration.",
    license="MIT",
)
